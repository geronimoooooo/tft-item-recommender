# TFT Item Recommender

A recommender system for items in Teamfight Tactics (Set 17),
trained on Master tier ranked matches via Riot API.

## Result

### Baseline comparison (trained and evaluated on fresh post-patch data)

| Recommender | Recall@3 | Precision@3 |
|---|---|---|
| Random (baseline) | 0.012 | 0.011 |
| Popularity | 0.131 | 0.114 |
| Frequency (per-champion) | 0.388 | 0.339 |
| Frequency + tier | 0.384 | 0.335 |
| Frequency + trait | **0.398** | **0.345** |

### Distribution shift testing

After collecting new data post-patch, I tested model robustness across 
data distributions. **Best configuration: Frequency + tier trained on 
combined (old + new) data, evaluated on new test set = 0.4246 Recall@3** 
(+4.6% vs single-distribution baseline).

## Project Story

Originally planned as augment recommender, pivoted to item recommender 
after discovering Riot API for Set 17 doesn't return augment data. 
Built incrementally with measurable improvements at each iteration:

- **Iteration 1:** Random / Popularity / Frequency baselines
- **Iteration 2:** Added tier feature (negative result)
- **Iteration 3:** Added traits feature (+2.3% Recall@3)
- **Distribution shift testing:** Combined training improves robustness

## Key Findings

I conducted distribution shift testing after the patch. The main insight: 
combined training (old + new data) provides a more robust model than 
training on just one dataset.

Frequency + tier model reached Recall@3 = 0.4246 on Combined → New, 
which is +4.6% relative to New → New baseline.

This contradicts the initial hypothesis that the old data would introduce 
noise in the model due to distribution shift. The reality is that the 
meta in TFT changes gradually, and old data complements fresh patterns 
rather than contradicting them.

Practical implication: for production, I would recommend not discarding 
pre-patch data, but supplementing post-patch for robustness. Sparsity 
issues that previously broke the tier feature are solved by increasing 
the sample size.


## Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, pandas
- **Data source:** Riot API (custom client with retries + checkpointing)
- **Dataset:** 731 Master tier matches across two collection windows 
  (April 28: 195 matches, May 5: 536 matches), Set 17 RU server

## Project Structure

```
tft-item-recommender/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 00_data_discovery.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── context.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── riot_client.py
│   └── recommenders/
│       ├── __init__.py
│       ├── frequency.py
│       ├── tier_frequency.py
│       ├── trait_recommender.py
│       ├── popularity_recommender.py
│       └── random_recommender.py
├── scripts/
│   ├── fetch_challenger_data.py
│   ├── distribution_combine_test.py
│   └── train_baseline.py
├── README.md
└── requirements.txt
```

## Iterations Log

| Iteration | Description | Recall@3 | vs prev |
|---|---|---|---|
| I1 | Frequency baseline (per-champion) | 0.3363 | — |
| I2 | Added tier feature | 0.3316 | -0.005 |
| I3 | Added traits feature | 0.3591 | +0.023 |
| Distribution shift | Combined training | 0.4246 | +0.057 (best) |
| I4 (planned) | Hybrid recommender | — | — |

## Setup

````bash
git clone https://github.com/Geronimoooooo/tft-augment-recommender.git
cd tft-augment-recommender

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

echo "RIOT_API_KEY=your_key" > .env
````

## Usage

````bash
python -m scripts.fetch_challenger_data       # collect data
python -m scripts.train_baseline              # train + evaluate
python -m scripts.distribution_combine_test   # distribution shift testing
````
