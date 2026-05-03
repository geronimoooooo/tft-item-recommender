"""Evaluation metrics for recommender systems."""

from typing import Sequence
from src.context import Context
import pandas as pd

def recall_at_k(
        recommend: Sequence[str],
        relevant: Sequence[str],
        k: int
) -> float:
    """Compute Recall@K for a single prediction.
    
    Recall@K = (relevant items found in top-K recommendations) / |relevant|
    
    Args:
        recommended: List of recommended items (will be truncated to K).
        relevant: List of items that are actually relevant (ground truth).
        k: Number of top recommendations to evaluate.
    
    Returns:
        Recall score in [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(recommend[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)

def precision_at_k(
        recommend: Sequence[str],
        relevant: Sequence[str],
        k: int
) -> float:
    """Compute Precision@K for a single prediction.
    
    Precision@K = (relevant items in top-K) / K
    
    Args:
        recommended: List of recommended items (will be truncated to K).
        relevant: List of relevant items (ground truth).
        k: Number of top recommendations.
    
    Returns:
        Precision score in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = set(recommend[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / k

def evaluate_recommender(
        recommender,
        test_df: pd.DataFrame,
        k: int = 3,
        placement_threshold: int = 2,
) -> dict:
    """Evaluate a recommender on test data.
    
    Filters test_df to high-placement records, then for each unit
    computes recall@k and precision@k between predicted and actual items.
    
    Args:
        recommender: Object with .recommend(context: Context) method.
        test_df: DataFrame with character_id, placement, itemNames.
        k: Top-K to evaluate.
        placement_threshold: Only evaluate on records with placement <= this.
    
    Returns:
        Dict with mean_recall_at_k, mean_precision_at_k, n_evaluated.
    """
    filtered = test_df[test_df['placement'] <= placement_threshold]

    recalls = []
    precisions = []
    n_skipped = 0
    for _, row in filtered.iterrows():
        actual_items = row['itemNames']  
        if not actual_items:
            continue  # skip units with no items
        context = Context(
            champion=row['character_id'],
            tier=row['tier'],
        )
        recommended = recommender.recommend(context)
        if not recommended:
            n_skipped += 1
            continue  # cold start; skip for now
        
        recalls.append(recall_at_k(recommended, actual_items, k))
        precisions.append(precision_at_k(recommended, actual_items, k))
    
    return {
        f'mean_recall_at_{k}': sum(recalls) / len(recalls) if recalls else 0,
        f'mean_precision_at_{k}': sum(precisions) / len(precisions) if precisions else 0,
        'n_evaluated': len(recalls),
        'n_skipped_cold_start': n_skipped,
    }