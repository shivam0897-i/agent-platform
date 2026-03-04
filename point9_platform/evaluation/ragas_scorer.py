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

    # Using singleton
    scorer = get_ragas_scorer()
    results = scorer.score(query=..., context=[...], response=..., reference=...)

    # Or create custom instance
    scorer = RagasScorer(model="gemini/gemini-2.0-flash")

Install:
    pip install "ragas>=0.1.0,<0.3.0" datasets langchain-community

IMPORTANT NOTES:
    - RAGAS API Compatibility: This module supports ragas v0.1.x and v0.2.x.
      The ragas library has breaking API changes between major versions.
      We pin to >=0.1.0,<0.3.0 and handle both API styles internally.
      If you upgrade ragas, test this scorer before deploying.
    - RAGAS uses an LLM internally to compute metrics. Each call to
      ``score()`` makes multiple LLM requests (one per metric).
      Expect 4-8 LLM calls per evaluation depending on metrics enabled.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RagasScorer:
    """
    RAGAS evaluation scorer for retrieval-augmented generation.

    Handles:
    - Context Precision & Recall
    - Faithfulness (grounding)
    - Answer Relevancy
    - Graceful failure (returns None per metric, never crashes)
    """

    METRIC_KEYS = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]

    def __init__(self, model: Optional[str] = None):
        """
        Initialize RAGAS scorer.

        Args:
            model: LiteLLM model identifier for the RAGAS evaluator LLM.
                   Falls back to the platform default model if not set.
        """
        self.model = model or self._default_model()
        self._initialized = False

    def _initialize(self):
        """Lazy-import ragas and datasets."""
        if self._initialized:
            return

        try:
            import ragas  # noqa: F401
            import datasets  # noqa: F401
            self._initialized = True
        except ImportError:
            logger.warning(
                "ragas or datasets not installed. "
                "Run: pip install ragas datasets"
            )

    # ==================== Public API ====================

    def score(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Run all RAGAS metrics and return a flat dict of scores.

        Args:
            query: The user question / prompt.
            context: List of context passages fed to the agent.
            response: The agent's final answer.
            reference: Ground-truth answer (needed for context_recall).

        Returns:
            Dict with keys matching ``METRIC_KEYS``.
            Values are floats in [0, 1] or ``None`` if unavailable.
        """
        results = self._empty_results()
        self._initialize()

        if not self._initialized:
            return results

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
            logger.warning("Could not import ragas metrics.")
            return results

        # Build HF Dataset expected by ragas
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

            eval_kwargs: Dict[str, Any] = {
                "dataset": dataset,
                "metrics": metrics,
            }
            eval_kwargs.update(self._build_llm_kwargs())

            ragas_result = ragas_evaluate(**eval_kwargs)

            for key in results:
                val = ragas_result.get(key)
                if val is not None:
                    results[key] = round(float(val), 4)

        except Exception:
            logger.error("RAGAS evaluation failed", exc_info=True)

        return results

    # ==================== Helpers ====================

    def _empty_results(self) -> Dict[str, Optional[float]]:
        return {k: None for k in self.METRIC_KEYS}

    def _build_llm_kwargs(self) -> Dict[str, Any]:
        """
        Try to wrap LiteLLM for RAGAS. Returns empty dict on failure.

        Handles both ragas v0.1.x (LangchainLLMWrapper) and v0.2.x
        (LangchainLLMWrapper moved or renamed) API styles.
        """
        # Strategy 1: ragas v0.1.x — LangchainLLMWrapper + ChatLiteLLM
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.chat_models import ChatLiteLLM

            llm = LangchainLLMWrapper(ChatLiteLLM(model=self.model))
            return {"llm": llm}
        except (ImportError, Exception):
            pass

        # Strategy 2: ragas v0.2.x — may accept llm string directly
        try:
            import ragas
            version = getattr(ragas, "__version__", "0.1.0")
            major_minor = tuple(int(x) for x in version.split(".")[:2])
            if major_minor >= (0, 2):
                logger.debug(
                    "ragas v%s detected; relying on native LLM config.", version
                )
        except Exception:
            pass

        logger.debug("Could not inject LiteLLM into RAGAS; using defaults.")
        return {}

    @staticmethod
    def _default_model() -> str:
        try:
            from point9_platform.settings.user import UserSettings
            return UserSettings().DEFAULT_LLM_MODEL
        except Exception:
            return "gemini/gemini-2.0-flash"


# Singleton instance
_ragas_scorer: Optional[RagasScorer] = None


def get_ragas_scorer() -> RagasScorer:
    """Get or create RAGAS scorer singleton instance."""
    global _ragas_scorer
    if _ragas_scorer is None:
        _ragas_scorer = RagasScorer()
    return _ragas_scorer
