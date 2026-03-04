"""
HuggingFace Evaluate Scorer
============================

Computes text-quality metrics using the ``evaluate`` library.

Metrics:
    - rouge1              — Unigram overlap (F1)
    - rougeL              — Longest-common-subsequence overlap (F1)
    - bertscore_precision — Semantic precision via contextual embeddings
    - bertscore_recall    — Semantic recall
    - bertscore_f1        — Semantic F1

Usage:
    from point9_platform.evaluation.hf_scorer import HFScorer, get_hf_scorer

    # Using singleton
    scorer = get_hf_scorer()
    results = scorer.score(response="...", reference="...")

    # Or custom instance
    scorer = HFScorer(bertscore_model="microsoft/deberta-base")

Install:
    pip install evaluate rouge_score bert_score

IMPORTANT NOTES:
    - BERTScore Model Download: The first call to ``score()`` will download
      the BERTScore model (~260 MB for distilbert-base-uncased). This is a
      one-time cost cached in HuggingFace's local cache directory.
      On HF Spaces or Docker, pre-download in the Dockerfile to avoid
      cold-start latency:
          RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
    - ROUGE is lightweight and has no model download.
    - Both metrics require a ground-truth ``reference`` answer.
      If no reference is provided, the Evaluator skips this scorer
      and returns None for all HF metrics.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HFScorer:
    """
    HuggingFace ``evaluate`` scorer for text-quality metrics.

    Handles:
    - ROUGE-1 and ROUGE-L
    - BERTScore (precision / recall / F1)
    - Graceful failure (returns None per metric, never crashes)
    """

    METRIC_KEYS = [
        "rouge1",
        "rougeL",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]

    def __init__(self, bertscore_model: str = "distilbert-base-uncased"):
        """
        Initialize HF scorer.

        Args:
            bertscore_model: HuggingFace model for BERTScore
                             (default: distilbert-base-uncased).
        """
        self.bertscore_model = bertscore_model

    # ==================== Public API ====================

    def score(
        self,
        response: str,
        reference: str,
    ) -> Dict[str, Optional[float]]:
        """
        Compute ROUGE and BERTScore between *response* and *reference*.

        Args:
            response: The agent's generated answer.
            reference: The ground-truth / expected answer.

        Returns:
            Dict with keys matching ``METRIC_KEYS``.
            Values are floats in [0, 1] or ``None`` on failure.
        """
        results = self._empty_results()

        results.update(self._compute_rouge(response, reference))
        results.update(self._compute_bertscore(response, reference))

        return results

    # ==================== ROUGE ====================

    def _compute_rouge(
        self,
        response: str,
        reference: str,
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {"rouge1": None, "rougeL": None}
        try:
            import evaluate

            rouge = evaluate.load("rouge")
            raw = rouge.compute(
                predictions=[response],
                references=[reference],
                use_aggregator=True,
            )
            if raw:
                out["rouge1"] = round(float(raw.get("rouge1", 0)), 4)
                out["rougeL"] = round(float(raw.get("rougeL", 0)), 4)
        except ImportError:
            logger.warning(
                "evaluate or rouge_score not installed. "
                "Run: pip install evaluate rouge_score"
            )
        except Exception:
            logger.error("ROUGE computation failed", exc_info=True)

        return out

    # ==================== BERTScore ====================

    def _compute_bertscore(
        self,
        response: str,
        reference: str,
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }
        try:
            import evaluate

            bertscore = evaluate.load("bertscore")
            raw = bertscore.compute(
                predictions=[response],
                references=[reference],
                model_type=self.bertscore_model,
            )
            if raw:
                out["bertscore_precision"] = round(float(raw["precision"][0]), 4)
                out["bertscore_recall"] = round(float(raw["recall"][0]), 4)
                out["bertscore_f1"] = round(float(raw["f1"][0]), 4)
        except ImportError:
            logger.warning(
                "evaluate or bert_score not installed. "
                "Run: pip install evaluate bert_score"
            )
        except Exception:
            logger.error("BERTScore computation failed", exc_info=True)

        return out

    # ==================== Helpers ====================

    def _empty_results(self) -> Dict[str, Optional[float]]:
        return {k: None for k in self.METRIC_KEYS}


# Singleton instance
_hf_scorer: Optional[HFScorer] = None


def get_hf_scorer() -> HFScorer:
    """Get or create HF scorer singleton instance."""
    global _hf_scorer
    if _hf_scorer is None:
        _hf_scorer = HFScorer()
    return _hf_scorer
