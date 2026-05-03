"""
For system clarity we need to understand the bottom line of predict ability with random recommender.

For usage: python -m random_recommender.py
"""

import pandas as pd
import random as random_module
from src.context import Context

class RandomRecommender:
    """Recommend K random items from the overall pool of items."""
    def __init__(self,
                 top_k: int = 3,
                 random_state: int = 42,):
        
        """Initialize the recommender."""
        
        self.top_k = top_k
        self.random_state = random_state
        self.rng = random_module.Random(random_state)
        self.all_items: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:

        """Learn the overall pool of items from the data."""

        unique_items = set()
        for items in df['itemNames']:
            for item in items:
                unique_items.add(item)
        self.all_items = list(unique_items)

    def recommend(self, context: Context) -> list[str]:

        """Recommend K random items from the overall pool, ignoring champion."""

        if not self.all_items:
            return []
        elif len(self.all_items) <= self.top_k:
            return list(self.all_items)
        else:
            return self.rng.sample(self.all_items, self.top_k)