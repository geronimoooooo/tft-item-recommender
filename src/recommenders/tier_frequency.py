"""Tier-aware frequency recommender."""

import pandas as pd
from collections import Counter

from src.context import Context
from src.recommenders.frequency import FrequencyRecommender


class TierFrequencyRecommender(FrequencyRecommender):
    """Frequency recommender grouped by (champion, tier).
    
    Falls back to champion-only frequency if a (champion, tier) 
    pair was unseen during training.
    """
    def __init__(self,
                 top_k: int = 3,
                 placement_threshold: int = 2) -> None:
        super().__init__(top_k=top_k, placement_threshold=placement_threshold)
        self.champion_tier_items: dict[tuple[str, int], list[str]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Learn champion and tier → top-K items mapping from data.
        
        Args:
            df: DataFrame with columns 'character_id', 'num_items', 'placement', 'itemNames', 'tier'.
        """
        super().fit(df)

        filtered = df[
            (df['placement'] <= self.placement_threshold) &
            (df['num_items'] > 0)
        ]

        champion_tier_counters = {}
        for (champion, tier), group in filtered.groupby(['character_id', 'tier']):
            all_items = []
            for items_list in group['itemNames']:
                all_items.extend(items_list)
            champion_tier_counters[(champion, tier)] = Counter(all_items)

        self.champion_tier_items = {
            key: [item for item, _ in counter.most_common(self.top_k)]
            for key, counter in champion_tier_counters.items()
        }
    
    def recommend(self, context: Context) -> list[str]:
        """Return top-K items for a given champion and champion's tier.
        
        Args:
            champion: Champion ID (e.g., 'TFT17_Teemo').
            tier: int (e.g. "2")
        
        Returns:
            List of K item IDs. Empty if champion is unknown (cold start).
            If model don't know due to train df champion's tier, use parent recommend meethod and returns only champion's recomendation.
        """
        key = (context.champion, context.tier)
        if key in self.champion_tier_items:
            return self.champion_tier_items[key]
        
        return super().recommend(context)
    
    def __repr__(self) -> str:
        n = len(self.champion_tier_items)
        return f'TierFrequencyRecommender(top_k={self.top_k}, fitted_on={n}_pairs'