"""Context object for item recommendation.

A frozen dataclass representing the input context for any recommender.
Fields are added incrementally across iterations:
- Iteration 1: champion only
- Iteration 2: + tier
- Iteration 3: + traits (planned)
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Context:
    """Input context for recommending items.
    
    Attributes:
        champion: Champion ID (e.g., 'TFT17_Teemo').
        tier: Champion star level (1, 2, or 3). Default 1.
    """
    champion: str
    tier: int = 1