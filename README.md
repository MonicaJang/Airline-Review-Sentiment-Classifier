# ✈️ What Makes Passenger Recommend an Airline?
---

## 📋 Overview

This study investigates how different service feature representations of passenger reviews affect the predictive performance of recommendation outcomes. Specifically, it compares five approaches:

- **Numerical Ratings Only (Set A):** Predicting recommendation using only the structured numerical sub-ratings provided by passengers.
- **Document-Level Sentiment (Set B):** Predicting recommendation using overall sentiment scores (VADER) extracted from the full review text, without distinguishing between service aspects.
- **Rule-Based Aspect Sentiment (Set C):** Predicting recommendation using sentiment scores extracted for manually defined five service aspects (seat, food, staff, ground service, entertainment), identified through keyword matching and scored using VADER.
- **VADER + Rule-based Sentiment (Set D):** Combines the outputs of Set B and Set C. No additional preprocessing required as each component follows its respective pipeline.
- **Deep Learning-Based Aspect Sentiment (Set E):** Predicting recommendation using aspect-level sentiment scores extracted via a pre-trained Aspect-Based Sentiment Analysis (ABSA) BERT model.

The central research question is whether enriching feature representations leads to meaningful improvements in predicting passenger recommendation decisions.

----

## 🎯 Expected Key Insights

- Which feature set is most predictive of recommendation outcomes?
- Which service aspects have the greatest impact on whether a review recommends the service?
- How much does performance actually differ between VADER, Rule-based VADER, and BERT approaches?
- How do passenger preferences change over time?

## ⭐ Exploratory Insights (separate from the text analysis)

- Which service aspects saw the sharpest sentiment decline during COVID?
- Which aspects recovered fastest post-pandemic?
- How did aspect-level sentiment shift over the course of the COVID period?

  
---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | [Skytrax Airline Reviews](https://www.airlinequality.com) |
| **Kaggle Download Link** | [https://www.kaggle.com/datasets/juhibhojani/airline-reviews] |
| **Records** | 23,171 reviews |
| **Coverage** | 497 airlines, 2002–2023 |
| **Target Variable** | `recommended` (yes / no) |

---

## 🔍 Methodology

### 1. Service Aspect Identification

To enable a direct comparison between passenger-assigned numerical scores and text-derived sentiment, five key service dimensions were selected for aspect-level sentiment extraction (Seat, Food, Staff, Ground Service, and Entertainment). Theses were intentionally aligned with the 7 core numerical sub-ratings present in the Skytrax dataset (Seat Comfort, Cabin Staff Service, Food & Beverages, Ground Service, Inflight Entertainment, Wifi & Connectivity, Value For Money).

| Aspect | Keywords |
|---|---|
| 🪑 Seat | 'seat', 'legroom', 'comfort', 'recline', 'space', 'cushion', 'comfy', 'spacious', 'stretch' |
| 👩‍✈️ Staff | 'crew', 'staff', 'attendant', 'stewardess', 'service', 'friendly', 'rude', 'attentive' |
| 🍽️ Food | 'food', 'meal', 'drink', 'snack', 'beverage', 'dining', 'menu', 'vegetarian' |
| 🎬 Entertainment | 'entertainment', 'screen', 'movie', 'ife', 'wifi', 'music' |
| ⏱️ Ground Service | 'delay', 'late', 'on time', 'punctual', 'schedule', 'depart', 'cancel', 'luggage', 'suitcase', 'baggage', 'lost', 'checkin', 'check-in', 'refund', 'booking', 'boarding'| 

**ife stands for In-Flight Entertainment**


### 2. Text Pre-Processing

- **Set A (Numerical Only)**: This set relies on numerical sub-ratings as features. Hence, text pre-processing is not required.
- **Set B (VADER)**: Minimal preprocessing to preserve sentiment signals such as negations, punctuation emphasis (!, ?), and sentence structure. Stopword removal and tokenization are intentionally excluded as they may distort VADER's sentiment scoring.
- **Set C (Rule-based Aspect Sentiment)**: Full preprocessing pipeline applied to improve keyword matching accuracy. Contraction expansion is performed before stopword removal to ensure negation words (e.g., not) are retained as separate tokens.
- **Set D (VADER + Rule-based)**: Combines the outputs of Set B and Set C. No additional preprocessing required as each component follows its respective pipeline.
- **Set E (ABSA BERT)**: Minimal preprocessing similar to Set B. Stopword removal and lemmatization are excluded as BERT relies on full sentence context and handles subword tokenization (WordPiece) internally.

| Technique | Set B | Set C | Set E |
|---|---|---|---|
| Lowercasing | ✅ | ✅ | ✅ |
| HTML / Special Character Removal | ✅ | ✅ | ✅ |
| Whitespace Normalization | ✅ | ✅ | ✅ |
| Contraction Expansion | ✅ | ✅ (before stopword removal) | ✅ |
| Punctuation Removal (partial) | ✅ | ✅ | ✅ |
| Stopword Removal | ❌ | ✅ | ❌ |
| Tokenization | ❌ | ✅ | ❌ |
| Lemmatization | ❌ | ✅ | ❌ |


### 3. Sentiment Scoring

- **Document-Level:** VADER implementation on the full text
- **Rule-Based Aspect-Level:** Custom-curated keyword dictionaries mapped to specific aspects, scored via VADER
- **Deep Learning Aspect-Level:** Pre-trained ABSA BERT model


### 4. Feature Sets

| Set | Core Features | NaN in Core |
|---|---|---|
| A | Numerical sub-ratings | Yes → imputed in modelling pipeline |
| B | VADER document-level scores | None |
| C | Rule-based aspect VADER scores | Yes → not mentioned or insufficient tokens to compute a score; retained as NaN |
| D | B + C | Yes → aspect columns only; retained as NaN |
| E1 | ABSA-BERT full inference scores | None |
| E2 | ABSA-BERT keyword-gated scores | Yes → not mentioned; retained as NaN |

**Common Features for All Sets**: 
`Verified`, `Type Of Traveller`, `Seat Type`, `review_length`

> `Verified` reflects a self-selected user action (whether the reviewer chose to submit proof of travel) rather than an objective quality signal about the flight itself. It is retained as a predictive feature because the lower recommendation rate among verified reviewers holds consistently across most Traveller Type and Seat Type subgroups (with the exception of First Class, likely due to small 
sample size), suggesting it captures a reproducible reviewer behavioral tendency (e.g., a propensity toward more critical, effortful reviews) rather than actual service quality.

### 5. Classification Models
- Logistic Regression
- Random Forest
- XGBoost
- Other gradient boosting varaiants (potentially)


### 6. Interpretability
**SHAP values** are computed on the best-performing model to quantify each service dimension's contribution to the recommendation prediction.

### 7. COVID Control
A binary `covid_period` dummy variable (reviews from 2020–2022) is included to control for pandemic-related sentiment shifts.


---

## 🗂️ Repository Structure

```
Airline-Review-Sentiment-Classifier/
│
├── data/
│   ├── Airline_review_rawdata.csv (gitignored)
│   ├── 01_processed/
│   └── 02_final_sets/
│
├── src/
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_vader_sentiment.ipynb
│   ├── 04_aspect_sentiment.ipynb
│   ├── 05_absa_bert.ipynb
│   ├── 06_feature_sets.ipynb
│   └── 07_modeling.ipynb
│
├── results/
│   ├── figures/
│   └── metrics/
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
pandas, numpy, scikit-learn, xgboost
nltk, vaderSentiment, textblob
shap, matplotlib, seaborn
```


---

## 🚀 Quickstart

Notebooks are numbered to reflect the pipeline order — run them sequentially:

```
src/01_eda_cleaning.ipynb        # EDA & data cleaning
src/02_text_preprocessing.ipynb  # Text preprocessing
src/03_vader_sentiment.ipynb     # Document-level VADER sentiment (Set B)
src/04_aspect_sentiment.ipynb    # Rule-based aspect sentiment (Set C)
src/05_absa_bert.ipynb           # ABSA BERT aspect sentiment (Set E)
src/06_feature_sets.ipynb        # Feature set construction (A–E)
src/07_modeling.ipynb            # Model training, evaluation & SHAP
```

To reproduce the full pipeline, open and run each notebook in order via Jupyter Lab/Notebook:

```bash
jupyter lab
```

---

**Data source**: Airline reviews scraped from [airlinequality.com](https://www.airlinequality.com) via the publicly available Skytrax dataset. This project is for academic research purposes only.
