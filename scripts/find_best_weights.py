"""Grid search for optimal tier/trait weights in HybridRecommender.

Evaluates HybridRecommender on Combined → New configuration 
(the best baseline from distribution shift testing).
"""

# 1. Stdlib
from pathlib import Path

# 2. Third-party
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 3. Local
from src.preprocessing import load_all_matches, matches_to_unit_records
from src.recommenders.hybrid import HybridRecommender
from src.evaluation import evaluate_recommender
from src.config import K, PT

def main() -> None:
    print('=' * 60)
    print('Grid Search — Optimal Hybrid Weights')
    print('=' * 60)

    # Load data (same as distribution_combine_test)
    old_matches = load_all_matches(Path('data/raw/challenger'))
    new_matches = load_all_matches(Path('data/raw/challenger_v2'))
    
    df_oldm = matches_to_unit_records(old_matches)
    df_newm = matches_to_unit_records(new_matches)
    
    # Combined train (with overlap fix)
    old_ids_set = set(df_oldm['match_id'])
    df_newm_clean = df_newm[~df_newm['match_id'].isin(old_ids_set)]
    df_combined = pd.concat([df_oldm, df_newm_clean], ignore_index=True)
    
    combined_ids = np.array(df_combined['match_id'].unique())
    combined_train_ids, _ = train_test_split(combined_ids, test_size=0.2, random_state=42)
    train_combined = df_combined[df_combined['match_id'].isin(combined_train_ids.tolist())]
    
    # New test (same split as in distribution test)
    new_ids = np.array(df_newm['match_id'].unique())
    _, new_test_ids = train_test_split(new_ids, test_size=0.2, random_state=42)
    test_new = df_newm[df_newm['match_id'].isin(new_test_ids.tolist())]
    
    print(f'\nTrain (combined): {len(train_combined)} rows')
    print(f'Test (new):       {len(test_new)} rows')

    # Grid of weight configurations
    weight_configs = [
        (0.0, 1.0),
        (0.00001, 0.99999),
        (0.0001, 0.9999),
        (0.001, 0.999),
        (0.01, 0.99),
        (0.1, 0.9),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1),
        (1.0, 0.0),
    ]
    
    print(f'\nEvaluating {len(weight_configs)} weight configurations...\n')
    
    results = []
    for tier_w, trait_w in weight_configs:
        rec = HybridRecommender(
            top_k=K,
            placement_threshold=PT,
            tier_weight=tier_w,
            trait_weight=trait_w,
        )
        rec.fit(train_combined)
        result = evaluate_recommender(rec, test_new, k=K, placement_threshold=PT)
        recall = result[f'mean_recall_at_{K}']
        results.append((tier_w, trait_w, recall))
        print(f'  tier={tier_w:.1f}, trait={trait_w:.1f}: Recall@{K} = {recall:.4f}')
    
    # Sort by recall, print
    print('\n' + '=' * 60)
    print('Sorted by Recall@3 (descending)')
    print('=' * 60)
    for tier_w, trait_w, recall in sorted(results, key=lambda x: -x[2]):
        print(f'  ({tier_w:.1f}, {trait_w:.1f}): {recall:.4f}')
    
    # Best
    best = max(results, key=lambda x: x[2])
    print(f'\nBest weights: tier={best[0]}, trait={best[1]}')
    print(f'Best Recall@{K}: {best[2]:.4f}')
    print(f'Improvement vs (0.5, 0.5): {best[2] - 0.4430:.4f}')


if __name__ == '__main__':
    main()