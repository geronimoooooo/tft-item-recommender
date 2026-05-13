"""Hybrid recommender combining tier-aware and trait-aware models."""

import pandas as pd

from src.context import Context
from src.recommenders.trait_recommender import TraitFrequencyRecommender
from src.recommenders.tier_frequency import TierFrequencyRecommender

class HybridRecommender:
    """Hybrid recommender combining tier-aware and trait-aware models.
    
    Uses composition pattern: holds two sub-recommenders
    (TierFrequencyRecommender, TraitFrequencyRecommender) and 
    combines their normalized scores via weighted sum.
    
    Weights control the relative contribution of each signal:
    - tier_weight × tier_score + trait_weight × trait_score
    
    Items present in only one sub-recommender's output still
    contribute via their single weight.
    """

    def __init__(
            self,
            top_k: int = 3,
            placement_threshold: int = 2,
            tier_weight: float = 0.5,
            trait_weight: float = 0.5
    ) -> None:
        self.top_k = top_k
        self.placement_threshold = placement_threshold
        self.tier_weight = tier_weight
        self.trait_weight = trait_weight
        
        self.tier_recommender = TierFrequencyRecommender(
            top_k=top_k,
            placement_threshold=placement_threshold,
        )
        self.trait_recommender = TraitFrequencyRecommender(
            top_k=top_k,
            placement_threshold=placement_threshold,
        )
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit both sub-recommenders on the same data."""
        self.tier_recommender.fit(df)
        self.trait_recommender.fit(df)

    def score_all_items(self, context: Context) -> dict[str, float]:
        """Return weighted combined scores from tier and trait sub-recommenders.
        
        Args:
            context: Context with champion, tier, and traits.
        
        Returns:
            Dict of item → combined score. Empty if both sub-recommenders
            returned no scores.
        """
        tier_scores = self.tier_recommender.score_all_items(context)
        trait_scores = self.trait_recommender.score_all_items(context)

        combined = {}
        for item, score in tier_scores.items():
            combined[item] = score * self.tier_weight
        for item, score in trait_scores.items():
            combined[item] = combined.get(item, 0) + score * self.trait_weight
        
        return combined
    
    def recommend(self, context: Context) -> list[str]:
        """Return top-K items by combined score.
        
        Args:
            context: Context with champion, tier, and traits.
        
        Returns:
            List of K item IDs sorted by combined score (descending).
            Empty if neither sub-recommender has data.
        """
        scores = self.score_all_items(context)
        if not scores:
            return []
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return [item for item, _ in sorted_items[:self.top_k]]
    
    def __repr__(self) -> str:
        return (
        f'HybridRecommender('
        f'top_k={self.top_k}, '
        f'tier_weight={self.tier_weight}, '
        f'trait_weight={self.trait_weight})'
    )