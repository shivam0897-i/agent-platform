"""
Evaluation Module
=================

Reusable evaluation framework for any Point9 agent.

Provides three scoring backends:
    - RagasScorer: Context Precision, Context Recall, Faithfulness, Answer Relevance
    - HFScorer: ROUGE-1, ROUGE-L, BERTScore
    - LLMJudge: Hallucination, Self-Consistency, Content Safety

Usage:
    from point9_platform.evaluation import Evaluator, get_evaluator

    # Using singleton (runs all scorers)
    evaluator = get_evaluator()
    result = evaluator.evaluate(
        query="What is the refund policy?",
        context=["Our refund policy allows returns within 30 days."],
        response="You can return items within 30 days.",
        reference="Refund policy: 30-day return window.",
    )

    # Cherry-pick scorers
    evaluator = Evaluator(run_ragas=True, run_hf=False, run_llm_judge=False)

    # Async usage in FastAPI / async agents
    result = await evaluator.aevaluate(query=..., context=[...], response=...)

    # Use individual scorers directly
    from point9_platform.evaluation import RagasScorer, HFScorer, LLMJudge

IMPORTANT NOTES:
    - All eval dependencies are optional. Install with:
          pip install "point9-agent-platform[eval]"
    - RAGAS is pinned to >=0.4.0,<1.0.0 (updated for new API).
    - BERTScore downloads ~260MB model on first use. Pre-download in
      Docker to avoid cold-start latency.
    - LLMJudge makes N+2 LLM calls per evaluation (default 6 total).
    - get_evaluator() returns a shared singleton with all scorers enabled.
      Create Evaluator(...) for custom config.
    - aevaluate() wraps sync evaluation in asyncio.to_thread for async use.
"""

from point9_platform.evaluation.framework import Evaluator, EvaluationResult, get_evaluator
from point9_platform.evaluation.ragas_scorer import RagasScorer, get_ragas_scorer
from point9_platform.evaluation.hf_scorer import HFScorer, get_hf_scorer
from point9_platform.evaluation.llm_judge import LLMJudge, get_llm_judge

__all__ = [
    # Main evaluator
    "Evaluator",
    "EvaluationResult",
    "get_evaluator",
    # Individual scorers
    "RagasScorer",
    "get_ragas_scorer",
    "HFScorer",
    "get_hf_scorer",
    "LLMJudge",
    "get_llm_judge",
]
