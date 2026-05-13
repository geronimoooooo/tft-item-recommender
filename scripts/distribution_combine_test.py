'''
Train and evaluate a banch of recommenders with different data building approaches.

Old data: 28/04/2026, app. 195 matches
New data: 05/05/2026, app. 539 matches

Combined data: app. 734 matches

used models:
        'Frequency': FrequencyRecommender,
        'Frequency + tier': TierFrequencyRecommender,
        'Frequency + trait': TraitFrequencyRecommender,
        'Popularity': PopularityRecommender,
        'Random': RandomRecommender,

Usage:
    python -m scripts.distribution_combine_test
'''

# 1. Stdlib
from pathlib import Path

# 2. Third-party
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 3. Local imports
from src.preprocessing import load_all_matches, matches_to_unit_records
from src.recommenders.frequency import FrequencyRecommender
from src.recommenders.tier_frequency import TierFrequencyRecommender
from src.recommenders.random_recommender import RandomRecommender
from src.recommenders.popularity_recommender import PopularityRecommender
from src.recommenders.trait_recommender import TraitFrequencyRecommender
from src.recommenders.hybrid import HybridRecommender
from src.evaluation import evaluate_recommender
from src.config import K, PT

def main() -> None:
    print('=' * 60)
    print('TFT Item Recommender — New-Old Data Comparison with Recommenders')
    print('=' * 60)

    print('\n[1/4] Loading all matches...')
    old_matches = load_all_matches(Path('data/raw/challenger'))
    new_matches = load_all_matches(Path('data/raw/challenger_v2'))

    print('\n[2/4] Converting matches to DataFrames...')
    df_oldm = matches_to_unit_records(old_matches)
    df_newm = matches_to_unit_records(new_matches)

    print('\n[3/4] Splitting train/test (80/20 by match)...')
    old_match_ids = df_oldm['match_id'].unique()
    new_match_ids = df_newm['match_id'].unique()

    old_match_ids = np.array(old_match_ids)
    new_match_ids = np.array(new_match_ids)

    old_train_ids, old_test_ids = train_test_split(old_match_ids, test_size=0.2, random_state=42)
    new_train_ids, new_test_ids = train_test_split(new_match_ids, test_size=0.2, random_state=42)

    old_train_df = df_oldm[df_oldm['match_id'].isin(old_train_ids.tolist())]
    old_test_df = df_oldm[df_oldm['match_id'].isin(old_test_ids.tolist())]

    new_train_df = df_newm[df_newm['match_id'].isin(new_train_ids.tolist())]
    new_test_df = df_newm[df_newm['match_id'].isin(new_test_ids.tolist())]

    old_ids_set = set(df_oldm['match_id'])
    df_newm_clean = df_newm[~df_newm['match_id'].isin(old_ids_set)]

    print(f'  Overlap: {len(df_newm) - len(df_newm_clean)} rows from df_new dropped (duplicate match_ids)')

    df_combined = pd.concat([df_oldm, df_newm_clean], ignore_index=True)

    combined_match_ids = df_combined['match_id'].unique()
    combined_match_ids = np.array(combined_match_ids)

    combined_train_ids, combined_test_ids = train_test_split(combined_match_ids, test_size=0.2, random_state=42)
    combined_train_df = df_combined[df_combined['match_id'].isin(combined_train_ids.tolist())]
    combined_test_df = df_combined[df_combined['match_id'].isin(combined_test_ids.tolist())]

    print(f'  Old data (28.04.2026): Train: {len(old_train_df)} rows ({len(old_train_ids)} matches)')
    print(f'  Old data (28.04.2026): Test:  {len(old_test_df)} rows ({len(old_test_ids)} matches)')
    print('-'*60)
    print(f'  New data (05.05.2026): Train: {len(new_train_df)} rows ({len(new_train_ids)} matches)')
    print(f'  New data (05.05.2026): Test:  {len(new_test_df)} rows ({len(new_test_ids)} matches)')
    print('-'*60)
    print(f'  Combined data (05.05.2026): Train: {len(combined_train_df)} rows ({len(combined_train_ids)} matches)')
    print(f'  Combined data (05.05.2026): Test:  {len(combined_test_df)} rows ({len(combined_test_ids)} matches)')

    configs = [
        ('old -> old', old_train_df, old_test_df),
        ('new -> new', new_train_df, new_test_df),
        ('old -> new', old_train_df, new_test_df),
        ('new -> old', new_train_df, old_test_df),
        ('combined -> combined', combined_train_df, combined_test_df),
        ('combined -> new', combined_train_df, new_test_df),
        ('combined -> old', combined_train_df, old_test_df),
    ]

    print('\n[4/4] Training and evaluating...')
    recommenders = {
        'Frequency': FrequencyRecommender,
        'Frequency + tier': TierFrequencyRecommender,
        'Frequency + trait': TraitFrequencyRecommender,
        'Hybrid (30/70)': HybridRecommender,
        'Popularity': PopularityRecommender,
        'Random': RandomRecommender,
        }
    
    results = {}
    for config_name, train_df, test_df in configs:
        print(f'\n=== {config_name} ===')
        results[config_name] = {}

        for rec_name , RecClass in recommenders.items():
            if RecClass == RandomRecommender:
                rec = RecClass(top_k=K, random_state=42)
            elif RecClass == HybridRecommender:
                rec = RecClass(top_k=K, placement_threshold=PT, tier_weight=0.3, trait_weight=0.7)
            else:
                rec = RecClass(top_k=K, placement_threshold=PT)
            rec.fit(train_df)
            result = evaluate_recommender(rec, test_df, k=K, placement_threshold=PT)
            recall = result.get(f'mean_recall_at_{K}', 0)
            results[config_name][rec_name] = recall

            print(f'  {rec_name}: Recall@{K} = {recall:.4f}')
    
    print('\n' + '=' * 80)
    print('SUMMARY TABLE — Recall@3')
    print('=' * 80)

    header = f'{"Recommender":<30}'
    for config_name, _, _ in configs:
        header += f'{config_name:>20}'
    print(header)
    print('-' * 80)

    for rec_name in recommenders.keys():
        row = f'{rec_name:<30}'
        for config_name, _, _ in configs:
            recall = results[config_name][rec_name]
            row += f'{recall:>20.4f}'
        print(row)

if __name__ == '__main__':
    main()