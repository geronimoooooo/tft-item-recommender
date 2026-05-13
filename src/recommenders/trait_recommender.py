"""Trait-aware frequency recommender."""

from collections import Counter

import pandas as pd

from src.context import Context
from src.recommenders.frequency import FrequencyRecommender


class TraitFrequencyRecommender(FrequencyRecommender):
    """Frequency recommender grouped by (champion, single_trait) pairs.
    
    For each player, iterates over all active team traits and 
    aggregates item occurrences per (champion, trait) pair.
    
    On recommend, votes are collected from all traits in the team
    context: items present in top-K for multiple traits are favored.
    Falls back to champion-only frequency if no relevant data.
    """
    
    def __init__(
        self,
        top_k: int = 3,
        placement_threshold: int = 2,
    ) -> None:
        super().__init__(top_k=top_k, placement_threshold=placement_threshold)
        self.champion_trait_items: dict[tuple[str, str], list[str]] = {}
        self.champion_trait_scores: dict[tuple[str, str], dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Learn (champion, trait) → top-K items mapping from data.
        
        Args:
            df: DataFrame with columns 'character_id', 'placement', 
                'num_items', 'itemNames', 'traits'.
        """
        super().fit(df)

        filtered = df[
            (df['placement'] <= self.placement_threshold) &
            (df['num_items'] > 0)
        ]

        pair_counters: dict[tuple[str, str], Counter] = {}
        for _, row in filtered.iterrows():
            champion = row['character_id']
            items = row['itemNames']
            traits_team = row['traits']
            for trait in traits_team:
                key = (champion, trait)
                if key not in pair_counters:
                    pair_counters[key] = Counter()
                pair_counters[key].update(items)
        
        self.champion_trait_scores = {}
        for (champion, trait), counter in pair_counters.items():
            if not counter:
                continue
            max_count = max(counter.values())
            self.champion_trait_scores[(champion, trait)] = {
                item: count / max_count
                for item, count in counter.items()
            }

        self.champion_trait_items = {
            key: [item for item, _ in counter.most_common(self.top_k)]
            for key, counter in pair_counters.items()
        }
    
    def score_all_items(self, context: Context) -> dict[str, float]:
        """Return aggregated normalized scores across all context traits.

        For each trait in context, looks up (champion, trait) scores
        and sums them across all active team traits. Result is renormalized 
        to [0, 1] for scale consistency with other recommenders.

        Args:
        context: Context with champion and traits (set of trait names).

        Returns:
        Dict of item → aggregated score in [0, 1]. Falls back to 
        champion-only scores if traits are empty or no relevant data.
        """

        if not context.traits:
            return super().score_all_items(context)

        aggregated_scores = {}
        for trait in context.traits:
            key = (context.champion, trait)
            if key in self.champion_trait_scores:
                for item, score in self.champion_trait_scores[key].items():
                    aggregated_scores[item] = aggregated_scores.get(item, 0) + score
        
        if not aggregated_scores:
            return super().score_all_items(context)
        
        max_score = max(aggregated_scores.values())
        return {
            item: score / max_score
            for item, score in aggregated_scores.items()
            }
        

    def recommend(self, context: Context) -> list[str]:
        """Return top-K items aggregated across context traits.
        
        For each trait in context, looks up top-K items for 
        (context.champion, trait). Items appearing in top-K of 
        multiple traits get more votes.
        
        Args:
            context: Context with champion, tier, and traits.
        
        Returns:
            List of K item IDs. Falls back to champion-only 
            recommendation if traits are empty or no relevant data.
        """
        if not context.traits:
            return super().recommend(context)
        
        votes: Counter = Counter()
        for trait in context.traits:
            key = (context.champion, trait)
            if key in self.champion_trait_items:
                for item in self.champion_trait_items[key]:
                    votes[item] += 1
        
        if not votes:
            return super().recommend(context)
        
        return [item for item, _ in votes.most_common(self.top_k)]
    
    def __repr__(self) -> str:
        n = len(self.champion_trait_items)
        return f'TraitFrequencyRecommender(top_k={self.top_k}, fitted_on={n}_pairs)'