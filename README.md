# ✈️ What Makes Passengers Recommend an Airline?

---

## 📋 Overview

This study investigates how different service feature representations of passenger reviews affect the predictive performance of recommendation outcomes. Specifically, it compares six feature sets built from the same underlying reviews:

- **Numerical Ratings Only (Set A):** Predicting recommendation using only the structured numerical sub-ratings provided by passengers.
- **Document-Level Sentiment (Set B):** Predicting recommendation using overall sentiment scores (VADER) extracted from the full review text, without distinguishing between service aspects.
- **Rule-Based Aspect Sentiment (Set C):** Predicting recommendation using sentiment scores extracted for five manually defined service aspects (seat, food, staff, ground service, entertainment), identified through keyword matching and scored using VADER.
- **VADER + Rule-Based Sentiment (Set D):** Combines the outputs of Set B and Set C.
- **Deep Learning Aspect Sentiment — Full Inference (Set E1):** Aspect-level sentiment extracted via a pre-trained Aspect-Based Sentiment Analysis (ABSA) BERT model, run on every review regardless of keyword presence.
- **Deep Learning Aspect Sentiment — Keyword-Gated (Set E2):** Same ABSA-BERT model as E1, but only run when an aspect's keywords are present in the review (aspect otherwise left as missing).

The central research question is whether enriching feature representations leads to meaningful improvements in predicting passenger recommendation decisions, and which service aspects actually drive that prediction.

---

## 🎯 Key Research Questions

- Which feature set is most predictive of recommendation outcomes?
- Which service aspects have the greatest impact on whether a review recommends the service?
- How much does performance actually differ between numerical ratings, VADER, rule-based aspect sentiment, and BERT-based approaches?
- How much of numerical ratings' predictive power can text alone (BERT) recover?

## ⭐ Supplementary Exploratory Insights (separate from the main predictive pipeline)

`Airline Name` and a COVID-period indicator were deliberately **excluded from the main predictive pipeline** — a model conditioned on specific airline identities or a fixed pandemic window would not generalize to future, unseen reviews. These variables are instead the subject of a **separate, descriptive (non-predictive) supplementary notebook**:

- Which service aspects saw the sharpest sentiment decline during COVID?
- Which aspects recovered fastest post-pandemic?
- How did aspect-level sentiment shift over the course of the COVID period?
- How does overall passenger preference (which aspects matter most) shift over time?

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | [Skytrax Airline Reviews](https://www.airlinequality.com) |
| **Kaggle Download Link** | [https://www.kaggle.com/datasets/juhibhojani/airline-reviews] |
| **Records (after cleaning, used in modelling)** | 22,980 reviews |
| **Coverage** | Multiple airlines, 2002–2023 |
| **Target Variable** | `Recommended` (yes / no) |

---

## 🔍 Methodology

### 1. Service Aspect Identification

Five key service dimensions were selected for aspect-level sentiment extraction (Seat, Food, Staff, Ground Service, Entertainment), aligned as closely as possible with Skytrax's numerical sub-ratings.

| Aspect | Keywords |
|---|---|
| 🪑 Seat | 'seat', 'legroom', 'comfort', 'recline', 'space', 'cushion', 'comfy', 'spacious', 'stretch' |
| 👩‍✈️ Staff | 'crew', 'staff', 'attendant', 'stewardess', 'service', 'friendly', 'rude', 'attentive' |
| 🍽️ Food | 'food', 'meal', 'drink', 'snack', 'beverage', 'dining', 'menu', 'vegetarian' |
| 🎬 Entertainment | 'entertainment', 'screen', 'movie', 'ife', 'wifi', 'music' |
| ⏱️ Ground Service | 'delay', 'late', 'on time', 'punctual', 'schedule', 'depart', 'cancel', 'luggage', 'suitcase', 'baggage', 'lost', 'checkin', 'check-in', 'refund', 'booking', 'boarding' |

**ife stands for In-Flight Entertainment*

### 2. Text Pre-Processing

| Technique | Set B | Set C | Set E1/E2 |
|---|---|---|---|
| Lowercasing | ✅ | ✅ | ✅ |
| HTML / Special Character Removal | ✅ | ✅ | ✅ |
| Whitespace Normalization | ✅ | ✅ | ✅ |
| Contraction Expansion | ✅ | ✅ (before stopword removal) | ✅ |
| Punctuation Removal (partial) | ✅ | ✅ | ✅ |
| Stopword Removal | ❌ | ✅ | ❌ |
| Tokenization | ❌ | ✅ | ❌ |
| Lemmatization | ❌ | ✅ | ❌ |

- **Set B (VADER)**: Minimal preprocessing to preserve sentiment signals such as negations and punctuation emphasis. Stopword removal/tokenization are excluded, as they can distort VADER's scoring.
- **Set C (Rule-based Aspect Sentiment)**: Full preprocessing pipeline for keyword-matching accuracy. Contraction expansion happens before stopword removal so negation words are retained as separate tokens.
- **Set E1/E2 (ABSA BERT)**: Minimal preprocessing; BERT relies on full sentence context and handles subword tokenization (WordPiece) internally.

### 3. Sentiment Scoring

- **Document-Level:** VADER on the full review text.
- **Rule-Based Aspect-Level:** Custom keyword dictionaries mapped to aspects, scored via VADER.
- **Deep Learning Aspect-Level:** Pre-trained ABSA-BERT model, run in two modes (full inference vs. keyword-gated).

### 4. Feature Sets

| Set | Core Features | NaN in Core | Handling |
|---|---|---|---|
| A | 5 numerical sub-ratings | Yes | Imputed **inside the modelling pipeline** (train-fit `GroupMedianImputer`, grouped by `Type Of Traveller`) to prevent leakage |
| B | VADER document-level scores | None | — |
| C | Rule-based aspect VADER scores | Yes | Retained as `NaN`; handled per-model (native for tree models, 0-fill + missing indicator otherwise) |
| D | B + C | Yes (aspect columns only) | Same as C |
| E1 | ABSA-BERT full inference scores | None | — |
| E2 | ABSA-BERT keyword-gated scores | Yes | Same handling as C |

> **Note on Set A:** Skytrax provides 7 numerical sub-ratings, but `Inflight Entertainment` and `Wifi & Connectivity` were **dropped due to high missingness** during feature-set construction. Set A therefore uses 5 sub-ratings: `Seat Comfort`, `Cabin Staff Service`, `Food & Beverages`, `Ground Service`, `Value For Money`. This means Set A has no direct counterpart for the "Entertainment" aspect used in Sets C/D/E1/E2, and Sets C/D/E1/E2 have no counterpart for "Value For Money" — a constraint carried through to the cross-Set SHAP comparison (see Interpretability below).

**Common Features for All Sets**: `Verified`, `Type Of Traveller`, `Seat Type`, `review_length`

> `Verified` reflects a self-selected user action (whether the reviewer chose to submit proof of travel) rather than an objective quality signal about the flight itself. It is retained as a predictive feature because the lower recommendation rate among verified reviewers holds consistently across most Traveller Type and Seat Type subgroups, suggesting it captures a reproducible reviewer behavioral tendency rather than actual service quality.

### 5. Train / Test Split

A **chronological** split (80% earliest / 20% most recent, by `Review Date`) is used instead of a random split — the goal is to test generalization to *future* reviews, which a random split cannot honestly evaluate. The same `row_id`-based split is applied identically to all six Sets, so every Set is compared on exactly the same reviews.

This surfaces a **class-balance shift**: `Recommended` rate is 35.9% in train vs. 23.5% in test, driven by a genuine long-term downward trend in the data (not seasonality — checked and ruled out) rather than by the split boundary itself. This has two downstream consequences, both addressed explicitly:
- **ROC-AUC is used as the primary metric** for comparing Sets/models (relatively insensitive to prevalence shifts); PR-AUC is interpreted relative to each split's own baseline (its positive rate) rather than compared at face value.
- **Threshold calibration** (see below) corrects for the resulting overprediction of `Recommended` at the default 0.5 threshold.

### 6. Classification Models

- Logistic Regression
- Random Forest
- XGBoost
- **LightGBM**

All four are evaluated at default hyperparameters across all six Sets (24 combinations total) as a baseline. **LightGBM is then tuned** (`RandomizedSearchCV`, `TimeSeriesSplit` CV) and adopted as the single model used across all six Sets — chosen over per-Set "best model" selection to enable a fair, apples-to-apples Set comparison and consistent SHAP interpretation, since the performance differences between LightGBM and each Set's best alternative are within noise (≤0.003 ROC-AUC, confirmed via paired Wilcoxon tests across CV folds).

### 7. Threshold Calibration

Since predicted probabilities are implicitly anchored to the higher train-period base rate, the default threshold (0.5) causes overprediction of `Recommended` on test data. A calibrated decision threshold is selected **per Set**, using only train-side `TimeSeriesSplit` CV validation folds (the test set is never used to choose the threshold), then applied once to test. This reduces the predicted-vs-actual positive rate gap across all six Sets while keeping precision/recall balanced.

### 8. Interpretability

**SHAP values** (`TreeExplainer`, exact Shapley values) are computed on the final tuned LightGBM pipeline for **each of the six Sets**, on the held-out chronological test set. Because each Set uses a different feature engineering methodology, raw feature names differ across Sets (e.g. Set C's `aspect_seat` vs. Set E2's `absa_seat_e2`); a verified aspect mapping is used to compare relative importance (normalized rank) across Sets on a common scale. Set B has no aspect-level breakdown (VADER scores are document-level) and is included in per-Set analysis but excluded from the cross-Set aspect comparison.

### 9. COVID / Airline Control (Supplementary Only)

Earlier plans included a binary COVID-period dummy as a control feature in the main model. This was revised: `Airline Name` and a COVID-period indicator are **excluded from the main predictive pipeline** entirely, since a model conditioned on these would not generalize to future, unseen data. They are instead the subject of a separate, purely descriptive supplementary notebook (see Repository Structure).

---

## 🗂️ Repository Structure

```
Airline-Review-Sentiment-Classifier/
│
├── 1_data/
│   ├── Airline_review_rawdata.csv (gitignored)
│   ├── 01_processed/
│   └── 02_final_sets/
│
├── 2_src/
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_vader_sentiment.ipynb
│   ├── 04_aspect_sentiment.ipynb
│   ├── 05_absa_bert.ipynb
│   ├── 06_feature_sets.ipynb
│   ├── 07_modelling_pipeline.ipynb      # baseline: 4 models x 6 Sets, chronological split
│   ├── 08_hyperparameter_tuning.ipynb   # LightGBM tuning + threshold calibration
│   ├── 09_shap_analysis.ipynb           # SHAP, per-Set + cross-Set aspect comparison
│   ├── 10_covid_spinoff_analysis.ipynb  # supplementary, descriptive only (planned)
│   ├── modelling_utils.py               # shared pipeline/transformer logic
│   └── fitted_pipelines/                # saved .joblib models + calibrated threshold .json
│
├── 3_results/
│   ├── 01_metrics/
│   └── 02_figures/
│
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/MonicaJang/Airline-Review-Sentiment-Classifier.git
cd Airline-Review-Sentiment-Classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

```
# Key Dependencies
pandas, numpy, scikit-learn, xgboost, lightgbm
scipy, joblib
nltk, vaderSentiment, textblob
shap, matplotlib, seaborn
```

---

## 🚀 Quickstart

Notebooks are numbered to reflect the pipeline order — run them sequentially:

```
2_src/01_eda_cleaning.ipynb          # EDA & data cleaning
2_src/02_text_preprocessing.ipynb    # Text preprocessing
2_src/03_vader_sentiment.ipynb       # Document-level VADER sentiment (Set B)
2_src/04_aspect_sentiment.ipynb      # Rule-based aspect sentiment (Set C)
2_src/05_absa_bert.ipynb             # ABSA-BERT aspect sentiment (Sets E1/E2)
2_src/06_feature_sets.ipynb          # Feature set construction (A-E2)
2_src/07_modelling_pipeline.ipynb    # Baseline: 4 models x 6 Sets, chronological split
2_src/08_hyperparameter_tuning.ipynb # LightGBM tuning, threshold calibration
2_src/09_shap_analysis.ipynb         # SHAP interpretability, cross-Set comparison
```

To reproduce the full pipeline, open and run each notebook in order via Jupyter Lab/Notebook:

```bash
jupyter lab
```

---

## 📊 Headline Result

Across every check performed — 4 model families at default hyperparameters, 5-fold chronological CV, and after LightGBM hyperparameter tuning — the Set ranking is consistent:

**A (numeric ratings) > E1 (BERT, full inference) > E2 (BERT, keyword-gated) > D (VADER + rule-based) > B (VADER) > C (rule-based)**

Numeric ratings remain the strongest predictor, but BERT-based sentiment recovers most of that predictive power from text alone — relevant for deployment scenarios where numeric ratings aren't available. BERT-based methods consistently outperform rule-based/lexicon methods across every paired comparison tested.

---

**Data source**: Airline reviews scraped from [airlinequality.com](https://www.airlinequality.com) via the publicly available Skytrax dataset. This project is for academic research purposes only.
