"""Debug HybridRecommender — verify combined scoring works."""

from pathlib import Path

from src.context import Context
from src.preprocessing import load_all_matches, matches_to_unit_records
from src.recommenders.hybrid import HybridRecommender


def main() -> None:
    matches = load_all_matches(Path('data/raw/challenger'))
    df = matches_to_unit_records(matches)

    rec = HybridRecommender(
        top_k=3,
        placement_threshold=2,
        tier_weight=0.5,
        trait_weight=0.5,
    )
    rec.fit(df)
    print(rec)
    print()

    ctx = Context(
        champion='TFT17_Mordekaiser',
        tier=2,
        traits=('TFT17_Mecha', 'TFT17_FlexTrait', 'TFT17_HPTank'),
    )
    print(f'Context: {ctx}')
    print(f'Top-3 (recommend): {rec.recommend(ctx)}')
    
    scores = rec.score_all_items(ctx)
    print(f'Combined scores count: {len(scores)}')
    
    top_5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
    print('Top-5 by combined score:')
    for item, score in top_5:
        print(f'  {item}: {score:.3f}')


if __name__ == '__main__':
    main()