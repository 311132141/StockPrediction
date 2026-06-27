# Equity Direction Forecasting with Alternative Data and Supervised Representation Learning

An end-to-end machine-learning system that forecasts the next-day price direction of Apple Inc.
(AAPL) by fusing market, sentiment, macroeconomic, and search-interest data, and that validates
every result with leakage-free, walk-forward methodology before stress-testing it in a realistic
trading backtest.

This project was built to demonstrate the full lifecycle of an applied quantitative-ML problem:
multi-source data engineering, feature and representation learning, model development from
classical baselines through deep learning, and — most importantly — the disciplined validation
required to produce results that can be trusted.

---

## Technical Competencies Demonstrated

- **Data engineering** — Automated ingestion and alignment of five heterogeneous sources
  (market data, news sentiment, global news tone via Google BigQuery, macroeconomic series, and
  web-search trends), including frequency resampling, missing-value handling, and time-aligned
  merging.
- **Feature engineering** — 40+ technical indicators plus learned features from a supervised
  autoencoder (representation learning).
- **Machine learning** — Benchmarking of 10+ classifiers, gradient-boosted trees (XGBoost,
  LightGBM), and automated hyperparameter optimization with Hyperopt (Tree-structured Parzen
  Estimator).
- **Deep learning** — A multi-output supervised autoencoder–MLP in TensorFlow/Keras, adapted
  from a Kaggle competition-winning architecture.
- **Financial-ML rigor** — Purged and embargoed time-series cross-validation, walk-forward
  validation, macroeconomic publication-lag modeling, and systematic elimination of look-ahead
  bias.
- **Quantitative evaluation** — An event-driven backtester with transaction costs reporting
  Sharpe ratio, maximum drawdown, and win rate against a buy-and-hold benchmark.
- **Reproducible engineering** — Modular, reusable pipeline functions and persisted model
  artifacts for inference on new data.

---

## Table of Contents

- [Problem statement](#problem-statement)
- [Solution architecture](#solution-architecture)
- [Data sources](#data-sources)
- [Feature engineering](#feature-engineering)
- [Modeling approach](#modeling-approach)
- [Validation methodology](#validation-methodology)
- [Backtesting framework](#backtesting-framework)
- [Results and interpretation](#results-and-interpretation)
- [Repository structure](#repository-structure)
- [Technology stack](#technology-stack)
- [Setup and configuration](#setup-and-configuration)
- [Usage](#usage)
- [Key learnings and future work](#key-learnings-and-future-work)
- [Disclaimer](#disclaimer)

---

## Problem statement

- **Objective:** Predict whether AAPL will close higher the following trading day — a binary
  classification problem: `Target = 1 if Close[t+1] > Close[t] else 0`.
- **Horizon:** One trading day.
- **Sample:** Daily data from 2010–2025 (3,800+ trading days); class balance is approximately
  53% up / 47% down, so a majority-class baseline scores ~53% on the full sample.
- **Success criteria:** Out-of-sample classification quality (accuracy, precision, recall, F1,
  ROC-AUC) and, critically, economic significance measured through a transaction-cost-aware
  trading backtest.

## Solution architecture

```
 ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────┐
 │ Data sources │→ │   Feature     │→ │  Representation   │→ │   Model training    │
 │ (5 streams)  │  │  engineering  │  │  learning (AE)    │  │ (purged CV / WF)    │
 └──────────────┘  └───────────────┘  └──────────────────┘  └─────────┬───────────┘
                                                                       │
            ┌──────────────────────────────────────────────────────────┘
            ▼
 ┌────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────────┐
 │  Ensemble (AE-MLP  │→ │  Walk-forward, leak-  │→ │  Event-driven backtest vs.   │
 │      + XGBoost)    │  │  free predictions     │  │  buy-and-hold (costs, Sharpe)│
 └────────────────────┘  └──────────────────────┘  └─────────────────────────────┘
```

The pipeline was developed in stages, with each iteration tightening the validation methodology
and removing sources of look-ahead bias — a progression that is central to the project's findings.

## Data sources

All data is for AAPL and is retrieved programmatically.

| Stream | Source | Library | Signal |
|--------|--------|---------|--------|
| Market data | Yahoo Finance | `yfinance` | Daily Open, High, Low, Close, Volume |
| News sentiment | NewsAPI headlines scored with VADER | `requests`, `nltk` | Daily headline count and average compound sentiment |
| Global news tone | GDELT v2 Events (Google BigQuery) | `google-cloud-bigquery` | Daily average tone of worldwide "Apple" coverage |
| Macroeconomics | Federal Reserve (FRED) | `fredapi` | 10-Year Treasury yield, CPI, GDP, unemployment rate |
| Search interest | Google Trends | `pytrends` | "Apple" search interest over time |

Lower-frequency series (monthly CPI, quarterly GDP, weekly Trends) are resampled to a daily
frequency with forward-fill, and gaps in sentiment and macro data are imputed with
neutral/last-known values to maintain a continuous, aligned panel.

## Feature engineering

The engineered feature set spans several families:

- **Trend:** simple and exponential moving averages (5/10/20/50/200-day) and moving-average
  crossovers.
- **Momentum:** RSI, MACD (line, signal, histogram), Stochastic Oscillator (%K, %D), Rate of
  Change, and 10-day momentum.
- **Volatility:** Bollinger Bands (width and position), rolling return volatility, and intraday
  range.
- **Returns:** multi-horizon simple and log returns (1/5/10/20-day) and price-change measures.
- **Volume:** On-Balance Volume, volume change, and volume ratios.
- **Calendar:** day-of-week, month, and quarter.
- **Alternative data:** the merged sentiment, macro, and search-interest signals.
- **Learned features:** an 8-dimensional bottleneck representation produced by a supervised
  autoencoder.

A Random-Forest importance analysis combined with target-correlation filtering reduces the panel
to the ~30 most informative features for modeling.

## Modeling approach

**Phase 1 — Classical baselines.** A controlled benchmark of 10+ scikit-learn and
gradient-boosting classifiers (Logistic Regression, Random Forest, SVM, KNN, MLP, Naive Bayes,
Decision Tree, AdaBoost, Gradient Boosting, XGBoost, and LightGBM), with hyperparameter tuning
over time-series splits. These establish a reference point and a feature-importance baseline.

**Phase 2 — Supervised autoencoder–MLP.** A multi-output neural network (adapted from the
winning solution of the Jane Street Market Prediction competition) trained on a single graph with
three objectives:

1. a **decoder** that reconstructs the (noise-corrupted) input — a denoising-autoencoder
   regularizer;
2. an **auxiliary classifier** operating on the learned bottleneck; and
3. the **primary classifier** operating on the concatenation of the original and encoded features.

The architecture uses batch normalization, Gaussian-noise input corruption, swish activations,
dropout, and label smoothing. Training samples are weighted in proportion to the magnitude of
daily returns so that high-information days carry greater influence, and hyperparameters are
optimized with Hyperopt.

**Ensemble.** The autoencoder–MLP is combined with a walk-forward XGBoost model through
probability averaging, with fold-level predictions aggregated using a geometric weighting scheme.

## Validation methodology

Rigorous, leakage-free validation is the defining feature of this project. The following controls
are applied:

- **Purged, embargoed time-series cross-validation.** Custom splitters (in the style of *Advances
  in Financial Machine Learning*) group observations by trading day, insert a purge gap between
  training and test folds, and embargo observations immediately following each test window to
  prevent information bleed.
- **Walk-forward validation.** A rolling ~3-year training window and ~3-month test window, with
  the model retrained at each step and only ever predicting forward in time.
- **Publication-lag modeling.** Macroeconomic features are lagged to reflect real-world release
  delays (GDP by one quarter, others by roughly one month), so the model cannot observe data
  before it would have been available.
- **Strictly out-of-sample scaling.** Feature scalers are fit on training data only, within each
  fold.
- **Realistic execution assumptions.** Positions are taken on the *previous* day's signal, since
  a prediction made at the close can only be acted upon afterward.

## Backtesting framework

A long/flat strategy — hold the asset when the model predicts an up-day, hold cash otherwise — is
evaluated through an event-driven backtester that applies a 0.1% transaction cost on every change
in position and reports cumulative and annualized return, annualized volatility, Sharpe ratio,
maximum drawdown, and win rate, always benchmarked against passive buy-and-hold.

## Results and interpretation

### Phase 1 — single chronological split (baseline)

| Model | Accuracy | F1 | Net Return | Max Drawdown | Sharpe |
|-------|---------:|----:|-----------:|-------------:|-------:|
| Random Forest | 0.575 | 0.477 | +5.2% | −11.0% | +0.36 |
| Gradient Boosting | 0.524 | 0.477 | −8.0% | −16.5% | −0.25 |
| SVM | 0.533 | 0.428 | −5.4% | −19.7% | −0.14 |
| XGBoost | 0.505 | 0.467 | −18.5% | −28.3% | −0.78 |
| Logistic Regression | 0.481 | 0.375 | −23.6% | −27.0% | −1.19 |

*(Full ten-model table is available in `models/model_performance.json`.)*

### Walk-forward, leakage-free evaluation (2013–2025, 3,024 trading days)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|----:|
| Autoencoder–MLP | 0.506 | 0.532 | 0.557 | 0.544 |
| XGBoost | 0.502 | 0.532 | 0.491 | 0.511 |
| Ensemble (AE + XGB) | 0.506 | 0.537 | 0.488 | 0.511 |

**Backtest of the leakage-free strategy** (long/flat, 0.1% costs, $10,000 initial capital):

| Strategy | Final value | Total return | Max drawdown |
|----------|------------:|-------------:|-------------:|
| Model-based (walk-forward) | ~$55,800 | +458% | −40.3% |
| Buy-and-hold AAPL | ~$151,500 | +1,415% | −38.5% |

### Interpretation

The progression from Phase 1 to leakage-free walk-forward evaluation is the central result of this
project. Initial experiments using a single chronological split suggested that certain models
carried a modest predictive edge. When the identical approach was subjected to purged, embargoed,
walk-forward validation — with macroeconomic publication lags and strictly out-of-sample scaling —
accuracy converged to approximately 50% and the trading strategy underperformed a passive
benchmark.

This is the expected and methodologically correct outcome: reliably predicting the next-day
direction of a single, highly liquid large-cap equity from public data is consistent with the
weak-form efficient-market hypothesis. The principal deliverable is therefore not a profitable
signal but a disciplined, reproducible framework that does not mislead its author. The gap between
the two result sets quantifies precisely how much apparent performance is attributable to
look-ahead bias — and demonstrates the judgment to distinguish a genuine edge from an artifact of
flawed evaluation.

## Repository structure

```
StockPrediction/
├── README.md
├── phase1_baseline.ipynb        Phase 1: classical-ML benchmark and backtest
├── Phase1_secondtry.ipynb       Phase 1, modularized: 10-model comparison and artifact persistence
├── phase 2.ipynb                Alternative-data fusion + autoencoder-MLP + purged CV
├── stage2_secondtry.ipynb       Complete pipeline: ingest → engineer → AE → walk-forward → ensemble → backtest
├── Untitled2.ipynb              Reusable module form (training, inference, Hyperopt, backtesting)
├── environment_backup.yml       Conda environment export
├── data/                        Intermediate and output datasets (see below)
├── models/                      Persisted models, scaler, metrics, and feature list
└── test_data/                   Held-out sample datasets
```

Selected datasets in `data/`:

| File | Description |
|------|-------------|
| `merged_data.csv` | Market data joined with all alternative-data sources |
| `feature_engineered_data.csv` | Full engineered feature panel (~60 columns) |
| `final_modeling_data.csv` | Top-ranked features selected for modeling |
| `autoencoder_features.csv` | Feature panel augmented with learned bottleneck features |
| `walkforward_predictions_*.csv` | Per-model leakage-free predictions |
| `ensemble_predictions_fixed.csv` | Final ensemble predictions |
| `backtest_results_fixed.csv` | Strategy versus buy-and-hold equity and drawdown series |

## Technology stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data & numerics | pandas, NumPy |
| Classical ML | scikit-learn |
| Gradient boosting | XGBoost, LightGBM |
| Deep learning | TensorFlow / Keras |
| Hyperparameter optimization | Hyperopt (TPE) |
| Technical indicators | `ta`, custom pandas implementations |
| NLP / sentiment | NLTK (VADER) |
| Data acquisition | yfinance, fredapi, pytrends, Google Cloud BigQuery, NewsAPI |
| Visualization | Matplotlib, seaborn |

## Setup and configuration

The project targets **Python 3.12**. Core scientific packages are pinned in
`environment_backup.yml` (a Conda export); the deep-learning and data-acquisition libraries are
installed via pip.

```bash
# 1) Create the base scientific environment
conda env create -f environment_backup.yml

# 2) Install the runtime libraries used by the pipeline
pip install tensorflow xgboost lightgbm \
            yfinance ta nltk fredapi pytrends \
            google-cloud-bigquery hyperopt \
            scikit-learn pandas numpy matplotlib seaborn requests joblib scipy

# 3) Download the NLTK VADER lexicon (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

**Credentials.** The data-acquisition layer requires API access to NewsAPI, FRED, and Google
Cloud BigQuery (via Application Default Credentials). All keys should be provided through
environment variables and never committed to source control:

```bash
export NEWSAPI_KEY="…"
export FRED_API_KEY="…"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

The committed datasets allow the modeling and backtesting stages to be reproduced without
re-running data collection.

## Usage

The project is organized as a sequence of Jupyter notebooks.

1. Open `stage2_secondtry.ipynb` — the most complete pipeline — in Jupyter or VS Code.
2. Optionally run the data-acquisition cells (with credentials configured), or proceed directly
   from the provided datasets in `data/`.
3. Execute the feature-engineering, autoencoder, walk-forward training, ensembling, and
   backtesting stages in order; each writes its output to `data/` and renders diagnostic plots.
4. For programmatic use, `Untitled2.ipynb` exposes a modular API
   (`load_data`, `train_model`, `predict_with_model`, `backtest_strategy`) and a `main()` entry
   point, including a routine to score new data with the persisted ensemble.

## Key learnings and future work

**Key learnings**

- Validation design, not model selection, dominated the outcome. The single most impactful
  contribution was identifying and removing look-ahead bias; doing so changed the conclusions
  entirely.
- A negative or null result, established rigorously, is a meaningful and trustworthy outcome — and
  a more valuable one than an inflated result built on a leaky evaluation.
- Alternative data is only as useful as its historical depth and point-in-time accuracy; shallow
  or back-filled sentiment history materially limits its contribution.

**Future work**

- Reframe the target toward risk-adjusted, multi-day, or magnitude-based objectives, and explore
  an abstention strategy that trades only on high-conviction days.
- Extend the universe from a single name to a cross-sectional ranking problem across many equities,
  where machine-learning signals tend to be more robust than single-name direction forecasts.
- Consolidate the notebooks into a packaged `src/` library with automated tests, configuration
  management, and experiment tracking, backed by a fixed, point-in-time data snapshot.

## Disclaimer

This repository is intended for educational and research purposes only. It does not constitute
investment advice, and the models presented have no demonstrated edge over a passive buy-and-hold
benchmark.
