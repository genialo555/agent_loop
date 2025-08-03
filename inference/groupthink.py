"""Multi-threaded inference with GroupThink voting (KL divergence).
Currently a stub; will call the underlying model `n_threads` times and merge
answers by KL-based voting.
"""

from typing import List


def generate(instruction: str, n_threads: int = 4) -> str:
    """Return an answer aggregated from several model runs (stub)."""
    # TODO: parallel calls into Ollama + voting
    answers: List[str] = [f"[Stub] Answer {i+1} for: {instruction}" for i in range(n_threads)]
    return max(answers, key=len)  # placeholder aggregation
