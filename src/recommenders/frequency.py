"""Frequency-based recommender for TFT items.

Approach: for each champion, recommend the top-K items most frequently
used by high-placement players (top 2).
"""

from collections import Counter
from typing import Optional
from src.context import Context

import pandas as pd

class FrequencyRecommender:
    """Recommend items by frequency among top-placement players.
    
    For each champion, count how often each item appears across
    high-placement players (configurable threshold). Return top-K
    items by count.
    """
    def __init__(self, 
                 top_k: int = 3,
                 placement_threshold: int = 2) -> None:
        """Initialize the recommender.
        
        Args:
            top_k: Number of items to recommend per champion.
            placement_threshold: Use only data from players who placed this rank or better (e.g., 2 means top 1-2 only).
        """
        self.top_k = top_k
        self.placement_threshold = placement_threshold
        self.champion_items: dict[str, list[str]] = {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """Learn champion → top-K items mapping from data.
        
        Args:
            df: DataFrame with columns 'character_id', 'placement', 'itemNames'.
        """
        filtered = df[df['placement'] <= self.placement_threshold]
        champion_to_item_counter = {}
        for champion, group in filtered.groupby('character_id'):
            all_items = []
            for items_list in group['itemNames']:
                all_items.extend(items_list)
            champion_to_item_counter[champion] = Counter(all_items)
        
        self.champion_items = {
            champ: [item for item, _ in counter.most_common(self.top_k)]
            for champ, counter in champion_to_item_counter.items()
        }
    
    def recommend(self, context: Context) -> list[str]:
        """Return top-K items for a given champion.
        
        Args:
            champion: Champion ID (e.g., 'TFT17_Teemo').
        
        Returns:
            List of K item IDs. Empty if champion is unknown (cold start).
        """
        return self.champion_items.get(context.champion, [])
    
    def __repr__(self) -> str:
        n = len(self.champion_items)
        return f'FrequencyRecommender(top_k={self.top_k}, fitted_on={n}_champions)'