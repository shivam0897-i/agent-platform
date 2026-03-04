"""
Evaluation Framework
====================

Unified evaluator that orchestrates all scorers and returns structured results.

Usage:
    from point9_platform.evaluation import Evaluator, get_evaluator

    # Using singleton (runs all available scorers)
    evaluator = get_evaluator()
    result = evaluator.evaluate(
        query="What is the refund policy?",
        context=["Our policy allows returns within 30 days."],
        response="You can return items within 30 days.",
        reference="Refund policy: 30-day return window.",
    )

    # Or custom instance (cherry-pick scorers)
    evaluator = Evaluator(
        model="openai/gpt-4o",
        run_ragas=True,
        run_hf=False,
        run_llm_judge=True,
    )

    # Async usage (wraps sync scorers in a thread)
    result = await evaluator.aevaluate(query=..., context=[...], response=...)

    print(result.to_dict())
    print(result.flat_scores())

IMPORTANT NOTES:
    - Singleton Caveat: ``get_evaluator()`` returns a shared instance with
      default config (all scorers enabled). If you need custom config, create
      your own ``Evaluator(...)`` instance instead. The singleton is for the
      common "run everything" case.
    - Async Support: ``aevaluate()`` runs the sync ``evaluate()`` in a
      background thread via ``asyncio.to_thread``. This is safe because
      each evaluation is independent and the underlying libraries (ragas,
      evaluate, litellm) are thread-safe for separate calls.
    - Total LLM Calls: With all scorers enabled, a single ``evaluate()``
      call makes ~10-14 LLM requests (RAGAS ~4-8, LLM Judge 6).
      Disable unused scorers to reduce cost and latency.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------- #

@dataclass
class EvaluationResult:
    """
    Typed wrapper around evaluation output.

    Attributes:
        ragas:      RAGAS metric scores (or empty dict if skipped).
        hf:         HuggingFace evaluate scores (or empty dict).
        llm_judge:  LLM-as-Judge scores (or empty dict).
        meta:       Timing and configuration metadata.
    """

    ragas: Dict[str, Optional[float]] = field(default_factory=dict)
    hf: Dict[str, Optional[float]] = field(default_factory=dict)
    llm_judge: Dict[str, Optional[float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten into a plain dict for serialisation / logging."""
        return {
            "ragas": self.ragas,
            "hf": self.hf,
            "llm_judge": self.llm_judge,
            "meta": self.meta,
        }

    def flat_scores(self) -> Dict[str, Optional[float]]:
        """
        Return every metric in a single-level dict.

        Keys are prefixed by scorer name to avoid collisions:
        ``ragas_context_precision``, ``hf_rouge1``,
        ``llm_judge_hallucination_score``, etc.
        """
        flat: Dict[str, Optional[float]] = {}
        for prefix, scores in [
            ("ragas", self.ragas),
            ("hf", self.hf),
            ("llm_judge", self.llm_judge),
        ]:
            for k, v in scores.items():
                flat[f"{prefix}_{k}"] = v
        return flat


# --------------------------------------------------------------------- #
# Evaluator class
# --------------------------------------------------------------------- #

class Evaluator:
    """
    Unified evaluator for Point9 agent responses.

    Orchestrates:
    - RagasScorer (RAGAS metrics)
    - HFScorer (ROUGE + BERTScore)
    - LLMJudge (Hallucination, Self-Consistency, Content Safety)

    Each scorer is optional — disabled scorers are silently skipped.
    Failed scorers return ``None`` per metric and never crash the caller.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        run_ragas: bool = True,
        run_hf: bool = True,
        run_llm_judge: bool = True,
        consistency_runs: int = 3,
        bertscore_model: str = "distilbert-base-uncased",
        timeout: int = 60,
    ):
        """
        Initialize the evaluator.

        Args:
            model: LiteLLM model id for RAGAS & LLM judge.
                   Falls back to the platform default model.
            run_ragas: Enable RAGAS scorer (default True).
            run_hf: Enable HuggingFace evaluate scorer (default True).
            run_llm_judge: Enable LLM-as-Judge scorer (default True).
            consistency_runs: Number of runs for self-consistency check.
            bertscore_model: HF model for BERTScore.
            timeout: LLM call timeout in seconds.
        """
        self.model = model
        self.run_ragas = run_ragas
        self.run_hf = run_hf
        self.run_llm_judge = run_llm_judge
        self.consistency_runs = consistency_runs
        self.bertscore_model = bertscore_model
        self.timeout = timeout

        # Lazy-init scorer instances on first use
        self._ragas_scorer = None
        self._hf_scorer = None
        self._llm_judge = None

    # ==================== Lazy Scorer Access ====================

    def _get_ragas_scorer(self):
        if self._ragas_scorer is None:
            from point9_platform.evaluation.ragas_scorer import RagasScorer
            self._ragas_scorer = RagasScorer(model=self.model)
        return self._ragas_scorer

    def _get_hf_scorer(self):
        if self._hf_scorer is None:
            from point9_platform.evaluation.hf_scorer import HFScorer
            self._hf_scorer = HFScorer(bertscore_model=self.bertscore_model)
        return self._hf_scorer

    def _get_llm_judge(self):
        if self._llm_judge is None:
            from point9_platform.evaluation.llm_judge import LLMJudge
            self._llm_judge = LLMJudge(
                model=self.model,
                consistency_runs=self.consistency_runs,
                timeout=self.timeout,
            )
        return self._llm_judge

    # ==================== Public API ====================

    def evaluate(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate an agent response across all enabled scorers.

        Args:
            query: The user question / prompt.
            context: List of context strings the agent had access to.
            response: The agent's generated answer.
            reference: Ground-truth answer (optional; required for
                       ROUGE, BERTScore, and RAGAS context_recall).

        Returns:
            An :class:`EvaluationResult` containing all computed scores.
        """
        start = time.time()
        result = EvaluationResult()

        # 1. RAGAS
        if self.run_ragas:
            result.ragas = self._run_ragas(query, context, response, reference)

        # 2. HuggingFace Evaluate (ROUGE + BERTScore)
        if self.run_hf:
            result.hf = self._run_hf(response, reference)

        # 3. LLM-as-Judge
        if self.run_llm_judge:
            result.llm_judge = self._run_llm_judge(query, context, response)

        # Metadata
        elapsed = round(time.time() - start, 2)
        result.meta = {
            "elapsed_seconds": elapsed,
            "scorers_run": [
                s for s, enabled in [
                    ("ragas", self.run_ragas),
                    ("hf", self.run_hf),
                    ("llm_judge", self.run_llm_judge),
                ] if enabled
            ],
            "model": self.model or "(default)",
        }

        logger.info("Evaluation complete in %ss", elapsed)
        return result

    async def aevaluate(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Async version of :meth:`evaluate`.

        Runs the synchronous evaluation in a background thread so it
        can be awaited in async agent pipelines and FastAPI endpoints
        without blocking the event loop.

        Args:
            Same as :meth:`evaluate`.

        Returns:
            An :class:`EvaluationResult` containing all computed scores.
        """
        return await asyncio.to_thread(
            self.evaluate,
            query=query,
            context=context,
            response=response,
            reference=reference,
        )

    # ==================== Internal Runners ====================

    def _run_ragas(
        self,
        query: str,
        context: List[str],
        response: str,
        reference: Optional[str],
    ) -> Dict[str, Optional[float]]:
        logger.info("Running RAGAS scorer …")
        try:
            return self._get_ragas_scorer().score(
                query=query,
                context=context,
                response=response,
                reference=reference,
            )
        except Exception:
            logger.error("RAGAS scorer crashed", exc_info=True)
            from point9_platform.evaluation.ragas_scorer import RagasScorer
            return {k: None for k in RagasScorer.METRIC_KEYS}

    def _run_hf(
        self,
        response: str,
        reference: Optional[str],
    ) -> Dict[str, Optional[float]]:
        if reference is None:
            logger.info("Skipping HF scorer — no reference answer provided.")
            from point9_platform.evaluation.hf_scorer import HFScorer
            return {k: None for k in HFScorer.METRIC_KEYS}

        logger.info("Running HuggingFace evaluate scorer …")
        try:
            return self._get_hf_scorer().score(
                response=response,
                reference=reference,
            )
        except Exception:
            logger.error("HF scorer crashed", exc_info=True)
            from point9_platform.evaluation.hf_scorer import HFScorer
            return {k: None for k in HFScorer.METRIC_KEYS}

    def _run_llm_judge(
        self,
        query: str,
        context: List[str],
        response: str,
    ) -> Dict[str, Optional[float]]:
        logger.info("Running LLM-as-Judge scorer …")
        try:
            return self._get_llm_judge().score(
                query=query,
                context=context,
                response=response,
            )
        except Exception:
            logger.error("LLM judge crashed", exc_info=True)
            from point9_platform.evaluation.llm_judge import LLMJudge
            return {k: None for k in LLMJudge.METRIC_KEYS}


# Singleton instance
_evaluator: Optional[Evaluator] = None


def get_evaluator() -> Evaluator:
    """Get or create Evaluator singleton instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator
