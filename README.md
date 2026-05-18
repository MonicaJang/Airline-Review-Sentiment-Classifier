# ✈️ What Drives Passenger Dissatisfaction?
### A Rule-Based Aspect Sentiment Analysis and Machine Learning Approach to Predicting Airline Recommendation


## 📋 Overview

This project investigates **which dimensions of airline service quality** — as expressed in unstructured passenger review text — most strongly predict whether a traveller would **recommend an airline**.

Rather than relying on structured sub-ratings (which suffer from up to **74.5% missing data** in this dataset), this study extracts **aspect-level sentiment scores** directly from review text using a rule-based NLP pipeline, then feeds those scores into machine learning classifiers to predict the binary recommendation outcome.

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | [Skytrax Airline Reviews](https://www.airlinequality.com) |
| **Kaggle Download Link** | [https://www.kaggle.com/datasets/juhibhojani/airline-reviews] |
| **Records** | 23,171 reviews |
| **Coverage** | 497 airlines, 2002–2023 |
| **Target Variable** | `recommended` (yes / no) |

> ⚠️ Structured sub-rating fields have substantial missingness (up to 74.5% for certain fields), which motivates the text-based aspect sentiment approach.

---

## 🔍 Methodology

### 1. Aspect Identification
Sentences in each review are matched against **dimension-specific keyword dictionaries** to identify which service aspect is being discussed.

| Aspect | Example Keywords |
|---|---|
| 🪑 Seat / Comfort | `seat`, `legroom`, `comfort`, `recline`, `space` |
| 👩‍✈️ Cabin Crew / Staff | `crew`, `staff`, `attendant`, `service`, `flight attendant` |
| 🍽️ Food & Beverage | `food`, `meal`, `drink`, `snack`, `beverage` |
| 🎬 Inflight Entertainment | `entertainment`, `IFE`, `screen`, `movie`, `wifi` |
| ⏱️ Operational Punctuality | `delay`, `on time`, `punctual`, `departure`, `schedule` |

### 2. Sentiment Scoring
Matched sentences are scored using two lexicon-based tools:
- **VADER** — optimized for short, informal, social-text sentiment
- **TextBlob** — polarity and subjectivity scores

Each review yields **per-aspect sentiment scores** (compound score aggregated across matched sentences).

### 3. Feature Sets (Ablation Study)

Three feature configurations are compared:

| Set | Features | Purpose |
|---|---|---|
| **(A) Baseline** | Available structured sub-ratings | Numerical benchmark |
| **(B) Document-level** | VADER sentiment on full review text | Coarse NLP baseline |
| **(C) Aspect-level** | 5 × aspect sentiment scores + COVID dummy | Full proposed approach |

### 4. Classification Models
- Logistic Regression
- Random Forest
- XGBoost

### 5. Interpretability
**SHAP values** are computed on the best-performing model to quantify each service dimension's contribution to the recommendation prediction.

### 6. COVID Control
A binary `covid_period` dummy variable (reviews from **2020–2022**) is included to control for pandemic-related sentiment shifts.

---

## 🗂️ Repository Structure

```
airline-review-sentiment-classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── 01_eda_cleaning.py
│   ├── 02_preprocessing.py
│   ├── 03_vader_sentiment.py
│   ├── 04_aspect_sentiment.py
│   ├── 05_feature_sets.py
│   ├── 06_modeling.py
│   └── 07_shap.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/airline-review-aspect-sentiment.git
cd airline-review-aspect-sentiment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
pandas, numpy, scikit-learn, xgboost
nltk, vaderSentiment, textblob
shap, matplotlib, seaborn
```

---

## 🚀 Quickstart

```python
from src.aspect_sentiment import AspectSentimentExtractor

extractor = AspectSentimentExtractor()
scores = extractor.extract("The crew was fantastic but the food was terrible and we arrived 2 hours late.")

# Output:
# {
#   'cabin_crew':   0.82,
#   'food':        -0.65,
#   'punctuality': -0.74,
#   'seat':         0.00,
#   'entertainment': 0.00
# }
```

Run the full pipeline:

```bash
python src/models.py --feature-set C --model xgboost --shap
```

---

## 📊 Expected Outputs

- **Ablation table**: Accuracy, F1, AUC across feature sets A / B / C
- **SHAP summary plot**: Feature importance ranked by mean |SHAP value|
- **Per-aspect sentiment distributions**: Violin plots by recommendation class
- **COVID period analysis**: Sentiment trend comparison pre / during / post pandemic

---

## 🧪 Ablation Study Design

```
Feature Set A  ──►  Logistic Regression  ─┐
Feature Set B  ──►  Random Forest        ─┼──► Compare AUC / F1
Feature Set C  ──►  XGBoost             ─┘
                        │
                        ▼
                  Best Model
                        │
                        ▼
                  SHAP Interpretation
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{airline_aspect_sentiment_2024,
  title  = {What Drives Passenger Dissatisfaction? A Rule-Based Aspect Sentiment Analysis
            and Machine Learning Approach to Predicting Airline Recommendation},
  year   = {2024},
  url    = {https://github.com/<your-username>/airline-review-aspect-sentiment}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> **Data source**: Airline reviews scraped from [airlinequality.com](https://www.airlinequality.com) via the publicly available Skytrax dataset. This project is for academic research purposes only.
