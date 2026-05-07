"""Frequency-based recommender for TFT items.

Approach: for each champion, recommend the top-K items most frequently
used by high-placement players (top 2).
"""

from collections import Counter
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
        self.champion_scores: dict[str, dict[str, float]] = {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """Learn champion → top-K items mapping from data.
        
        Args:
            df: DataFrame with columns 'character_id', 'placement', 'itemNames'.
        """
        filtered = df[(df['placement'] <= self.placement_threshold)
                       & (df['num_items'] > 0)]
        champion_to_item_counter = {}
        for champion, group in filtered.groupby('character_id'):
            all_items = []
            for items_list in group['itemNames']:
                all_items.extend(items_list)
            champion_to_item_counter[champion] = Counter(all_items)
        
        # Normalize counts to scores in [0, 1] using max-normalization
        self.champion_scores = {}
        for champion, counter in champion_to_item_counter.items():
            max_count = max(counter.values())
            self.champion_scores[champion] = {
                item: count / max_count
                for item, count in counter.items()
            }

        self.champion_items = {
            champ: [item for item, _ in counter.most_common(self.top_k)]
            for champ, counter in champion_to_item_counter.items()
        }
    
    def score_all_items(self, context: Context) -> dict[str, float]:
        """Return normalized scores for all items observed for this champion.
    
        Args:
            context: Context with champion information.
    
        Returns:
            Dict mapping item_id to score in [0, 1]. Empty if champion 
            is unknown (cold start).
        """
        return self.champion_scores.get(context.champion, {})

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