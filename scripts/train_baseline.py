"""Train and evaluate the frequency baseline recommender.

Usage:
    python -m scripts.train_baseline
"""
# 1. Stdlib
from pathlib import Path

# 2. Third-party
from sklearn.model_selection import train_test_split
import numpy as np

# 3. Local imports
from src.preprocessing import load_all_matches, matches_to_unit_records
from src.recommenders.frequency import FrequencyRecommender
from src.recommenders.tier_frequency import TierFrequencyRecommender
from src.recommenders.random_recommender import RandomRecommender
from src.recommenders.popularity_recommender import PopularityRecommender
from src.evaluation import evaluate_recommender
from src.config import K

def main() -> None:
    print('=' * 60)
    print('TFT Item Recommender — Baseline Comparison')
    print('=' * 60)

    print('\n[1/4] Loading matches...')
    matches = load_all_matches(Path('data/raw/challenger'))
    print(f'  Loaded {len(matches)} matches')

    print('\n[2/4] Converting to DataFrame...')
    df = matches_to_unit_records(matches)
    print(f'  Shape: {df.shape}')

    print('\n[3/4] Splitting train/test (80/20 by match)...')
    match_ids = df['match_id'].unique()
    match_ids = np.array(match_ids)
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
    train_df = df[df['match_id'].isin(train_ids.tolist())]
    test_df = df[df['match_id'].isin(test_ids.tolist())]
    print(f'  Train: {len(train_df)} rows ({len(train_ids)} matches)')
    print(f'  Test:  {len(test_df)} rows ({len(test_ids)} matches)')

    print('\n[4/4] Training and evaluating...')
    recommenders = {'Frequency (per-champion)': FrequencyRecommender(top_k=3, placement_threshold=2),
                    'Frequency + tier': TierFrequencyRecommender(top_k=K, placement_threshold=2),
                    'Popularity': PopularityRecommender(top_k=3, placement_threshold=2),
                    'Random': RandomRecommender(top_k=3, random_state=42)}
    results = {}
    for name, rec in recommenders.items():
        print(f'\n ---{name}---')
        rec.fit(train_df)
        result = evaluate_recommender(rec, test_df, k=K, placement_threshold=2)
        results[name] = result
    print('\n=== Results ===')
    print(f'{"Recommender":<30} {f"Recall@{K}":>10} {f"Precision@{K}":>13}')
    for name, result in results.items():
        recall = result.get(f'mean_recall_at_{K}', 0)
        precision = result.get(f'mean_precision_at_{K}', 0)
        print(f'{name:<30} {recall:>10.4f} {precision:>13.4f}')

if __name__ == '__main__':
    main()