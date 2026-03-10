"""
RAGAS Scorer
============

Computes retrieval-augmented generation metrics via the ``ragas`` library.

Metrics:
    - context_precision   — Are the retrieved contexts relevant to the query?
    - context_recall      — Does the context cover the reference answer?
    - faithfulness        — Is the response grounded in the provided context?
    - answer_relevancy    — Is the response relevant to the query?

Usage:
    from point9_platform.evaluation.ragas_scorer import RagasScorer, get_ragas_scorer

    scorer = get_ragas_scorer()
    results = scorer.score(query=..., context=[...], response=..., reference=...)

    # Custom model / custom embedding model
    scorer = RagasScorer(model="openai/gpt-4o", embedding_model="text-embedding-3-small")

Install:
    pip install "ragas>=0.4.0,<1.0.0" instructor datasets

Compatibility:
    - ragas >= 0.4.0 (PRIMARY): EvaluationDataset + SingleTurnSample +
      LiteLLMStructuredLLM via instructor.from_litellm. Metrics imported from
      ragas.metrics.collections. Results extracted via .to_pandas().
    - ragas < 0.4.0 (LEGACY): HuggingFace Dataset + LangchainLLMWrapper.
      Kept for users who cannot upgrade. Requires langchain-community.

IMPORTANT NOTES:
    - AnswerRelevancy requires an embedding model in addition to an LLM.
      It is included only when LiteLLMEmbeddings can be constructed.
      Set ``embedding_model`` on the scorer instance to control which
      embedding model is used (default: "text-embedding-ada-002").
    - Each call to ``score()`` makes multiple LLM requests (one per metric).
      Expect 3–4 LLM calls per evaluation (fewer if reference is omitted).
    - ragas uses its own async event loop internally. The scorer is safe to
      call from sync code; for async callers use Evaluator.aevaluate()
      which runs it in a thread via asyncio.to_thread.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_model_string(model: str) -> Tuple[str, str]:
    """Split 'provider/model-name' into (provider, model_name).

    Falls back to provider='openai' for bare model names like 'gpt-4o'.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return "openai", model


def _get_ragas_version() -> Tuple[int, int]:
    """Return (major, minor) of the installed ragas package."""
    try:
        import ragas
        parts = ragas.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 1)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class RagasScorer:
    """
    RAGAS evaluation scorer for retrieval-augmented generation.

    Handles:
    - Context Precision & Recall
    - Faithfulness (grounding)
    - Answer Relevancy (requires embeddings)
    - Graceful failure — returns ``None`` per metric, never raises
    """

    METRIC_KEYS = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
    ):
        """
        Args:
            model: LiteLLM model string (e.g. "gemini/gemini-2.0-flash",
                   "openai/gpt-4o"). Falls back to platform default.
            embedding_model: Embedding model used by AnswerRelevancy.
                             Must be a LiteLLM-supported embedding model.
        """
        self.model = model or self._default_model()
        self.embedding_model = embedding_model
        self._initialized = False
        self._ragas_version: Tuple[int, int] = (0, 1)

    def _initialize(self) -> None:
        if self._initialized:
            return
        try:
            import ragas  # noqa: F401
            self._ragas_version = _get_ragas_version()
            self._initialized = True
        except ImportError:
            logger.warning(
                "ragas not installed. "
                "Run: pip install 'ragas>=0.4.0,<1.0.0' instructor"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """Run all RAGAS metrics and return a flat dict of scores.

        Args:
            query:     The user question / prompt.
            context:   List of context passages fed to the agent.
            response:  The agent's final answer.
            reference: Ground-truth answer (enables context_recall).

        Returns:
            Dict with keys matching ``METRIC_KEYS``.
            Values are floats in [0, 1] or ``None`` if unavailable.
        """
        self._initialize()
        if not self._initialized:
            return self._empty_results()

        if self._ragas_version >= (0, 4):
            return self._score_v04(query, context, response, reference)
        return self._score_legacy(query, context, response, reference)

    # ------------------------------------------------------------------ #
    # ragas >= 0.4.x path
    # ------------------------------------------------------------------ #

    def _score_v04(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str],
    ) -> Dict[str, Optional[float]]:
        results = self._empty_results()

        try:
            from ragas import evaluate as ragas_evaluate, EvaluationDataset
            from ragas.dataset_schema import SingleTurnSample
            from ragas.metrics.collections import (
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                AnswerRelevancy,
            )
        except ImportError as exc:
            logger.warning("ragas 0.4.x imports unavailable: %s", exc)
            return results

        llm = self._build_llm_v04()
        if llm is None:
            logger.warning(
                "Could not initialise LLM for RAGAS 0.4.x; returning empty scores. "
                "Ensure 'instructor' is installed."
            )
            return results

        # Build sample using v0.4.x field names
        sample_kwargs: Dict[str, Any] = {
            "user_input": query,
            "retrieved_contexts": context,
            "response": response,
        }
        if reference is not None:
            sample_kwargs["reference"] = reference

        sample = SingleTurnSample(**sample_kwargs)
        dataset = EvaluationDataset(samples=[sample])

        # Metrics — all require llm; AnswerRelevancy also needs embeddings
        metrics: List[Any] = [
            ContextPrecision(llm=llm),
            Faithfulness(llm=llm),
        ]
        if reference is not None:
            metrics.append(ContextRecall(llm=llm))

        # AnswerRelevancy: best-effort, requires embeddings
        try:
            embeddings = self._build_embeddings_v04()
            if embeddings is not None:
                metrics.append(AnswerRelevancy(llm=llm, embeddings=embeddings))
        except Exception:
            logger.debug(
                "Skipping AnswerRelevancy: could not configure embeddings.",
                exc_info=True,
            )

        try:
            ragas_result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                raise_exceptions=False,
                show_progress=False,
            )
            row = ragas_result.to_pandas().iloc[0].to_dict()
            for key in self.METRIC_KEYS:
                val = row.get(key)
                if val is not None:
                    try:
                        fval = float(val)
                        if not math.isnan(fval):
                            results[key] = round(fval, 4)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            logger.error("RAGAS 0.4.x evaluation failed", exc_info=True)

        return results

    def _build_llm_v04(self) -> Optional[Any]:
        """Build a ``LiteLLMStructuredLLM`` for ragas 0.4.x.

        Uses ``instructor.from_litellm`` as the structured-output backend,
        which supports all 100+ LiteLLM providers without provider-specific
        client setup.
        """
        try:
            import instructor
            import litellm
            from ragas.llms import LiteLLMStructuredLLM

            client = instructor.from_litellm(litellm.completion)
            provider, model_name = _parse_model_string(self.model)
            return LiteLLMStructuredLLM(
                client=client,
                model=model_name,
                provider=provider,
            )
        except Exception:
            logger.warning(
                "Could not build LiteLLMStructuredLLM for ragas 0.4.x",
                exc_info=True,
            )
            return None

    def _build_embeddings_v04(self) -> Optional[Any]:
        """Build ``LiteLLMEmbeddings`` for the AnswerRelevancy metric."""
        try:
            from ragas.embeddings import LiteLLMEmbeddings
            return LiteLLMEmbeddings(model=self.embedding_model)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Legacy path — ragas < 0.4.x
    # ------------------------------------------------------------------ #

    def _score_legacy(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str],
    ) -> Dict[str, Optional[float]]:
        """Evaluation path for ragas < 0.4.  Uses HuggingFace Dataset API."""
        results = self._empty_results()

        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            from datasets import Dataset
        except ImportError:
            logger.warning("Could not import ragas metrics (legacy path).")
            return results

        data: Dict[str, Any] = {
            "question": [query],
            "contexts": [context],
            "answer": [response],
        }
        metrics = [context_precision, faithfulness, answer_relevancy]

        if reference is not None:
            data["ground_truth"] = [reference]
            metrics.append(context_recall)

        try:
            dataset = Dataset.from_dict(data)
            eval_kwargs: Dict[str, Any] = {"dataset": dataset, "metrics": metrics}
            eval_kwargs.update(self._build_llm_kwargs_legacy())
            ragas_result = ragas_evaluate(**eval_kwargs)
            for key in results:
                val = ragas_result.get(key)
                if val is not None:
                    results[key] = round(float(val), 4)
        except Exception:
            logger.error("RAGAS legacy evaluation failed", exc_info=True)

        return results

    def _build_llm_kwargs_legacy(self) -> Dict[str, Any]:
        """Wrap LiteLLM via LangchainLLMWrapper for ragas < 0.4."""
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.chat_models import ChatLiteLLM
            return {"llm": LangchainLLMWrapper(ChatLiteLLM(model=self.model))}
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _empty_results(self) -> Dict[str, Optional[float]]:
        return {k: None for k in self.METRIC_KEYS}

    @staticmethod
    def _default_model() -> str:
        try:
            from point9_platform.settings.user import UserSettings
            return UserSettings().DEFAULT_LLM_MODEL
        except Exception:
            return "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ragas_scorer: Optional[RagasScorer] = None


def get_ragas_scorer() -> RagasScorer:
    """Get or create the shared RAGAS scorer singleton."""
    global _ragas_scorer
    if _ragas_scorer is None:
        _ragas_scorer = RagasScorer()
    return _ragas_scorer
