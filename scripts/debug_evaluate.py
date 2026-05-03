"""Debug why FrequencyRecommender returns empty in evaluate_recommender."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.context import Context
from src.preprocessing import load_all_matches, matches_to_unit_records
from src.recommenders.frequency import FrequencyRecommender


def main() -> None:
    matches = load_all_matches(Path('data/raw/challenger'))
    df = matches_to_unit_records(matches)

    match_ids = np.array(df['match_id'].unique())
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
    train_df = df[df['match_id'].isin(train_ids.tolist())]
    test_df = df[df['match_id'].isin(test_ids.tolist())]

    rec = FrequencyRecommender(top_k=3, placement_threshold=2)
    rec.fit(train_df)

    filtered = test_df[test_df['placement'] <= 2]
    print(f'Total filtered: {len(filtered)}')

    for _, row in filtered.iterrows():
        if not row['itemNames']:
            continue

        print('=== First row with items ===')
        print(f'character_id: {row["character_id"]!r}')
        print(f'character_id type: {type(row["character_id"])}')
        print(f'tier: {row["tier"]!r}')
        print(f'tier type: {type(row["tier"])}')
        print(f'itemNames: {row["itemNames"]!r}')

        ctx = Context(champion=row['character_id'], tier=row['tier'])
        print(f'Context: {ctx!r}')

        result = rec.recommend(ctx)
        print(f'Result: {result!r}')
        print(f'Champion in trained dict? {row["character_id"] in rec.champion_items}')

        break


if __name__ == '__main__':
    main()