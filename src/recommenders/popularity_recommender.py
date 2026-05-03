"""
For system clarity we need to understand next line of predict ability with popularity recommender.

For usage: python -m popularity_recommender.py
"""

import pandas as pd
from collections import Counter
from src.context import Context

class PopularityRecommender:
    """Recommend the K most popular items globally among high-placement players.
    
    Ignores the champion entirely. The same K items are recommended for 
    every champion: those most frequently used by players who placed 
    within the configured threshold.
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
        self.popular_items: list[str] = []
    
    def fit(self, df: pd.DataFrame) -> None:
        """Learn overall popular items from data.
        
        Args:
            df: DataFrame with columns 'placement', 'itemNames'.
        """
        filtered = df[df['placement'] <= self.placement_threshold]

        if len(filtered) == 0:
            print('Warning: no data after placement filtering')
            self.popular_items = []
            return
        
        all_items = []
        for items_list in filtered['itemNames']:
            all_items.extend(items_list)
        
        item_counts = Counter(all_items)
        self.popular_items = [item for item, _ in item_counts.most_common(self.top_k)]
    
    def recommend(self, context: Context) -> list[str]:
        """Return top-K popular items (same for all champions).
        
        Args:
            champion: Champion ID (ignored in this model).
        
        Returns:
            List of K item IDs. Empty if no data was fitted.
        """
        return self.popular_items