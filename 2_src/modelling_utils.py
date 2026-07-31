"""
This is shared preprocessing pipeline logic for 07_modelling_pipeline, 08_hyperparameter_tuning, and 09_shap_analysis.

Custom transformer including GroupMedianImputer must live in an importable module, not redefined inline inside each notebook.
If the class is defined inside a notebook's __main__ namespace, 
a pipeline saved in one notebook's kernel cannot be unpickled in a different notebook's kernel. 
Importing from this shared file gives every notebook the same import path, so joblib.load() works everywhere.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TARGET = "Recommended"
CAT_COLS = ["Type Of Traveller", "Seat Type"]
ID_COL = "row_id"
NON_FEATURE_COLS = [ID_COL, "Review Date", "Date Flown", TARGET]

RATING_COLS = ["Seat Comfort", "Cabin Staff Service", "Food & Beverages",
               "Ground Service", "Value For Money"]

SET_SCHEMA = {
    "A":  dict(nan_numeric=[], set_a_rating_cols=RATING_COLS),
    "B":  dict(nan_numeric=[], set_a_rating_cols=None),
    "C":  dict(nan_numeric=["aspect_seat", "aspect_food", "aspect_staff",
                             "aspect_ground_service", "aspect_entertainment"],
               set_a_rating_cols=None),
    "D":  dict(nan_numeric=["aspect_seat", "aspect_food", "aspect_staff",
                             "aspect_ground_service", "aspect_entertainment"],
               set_a_rating_cols=None),
    "E1": dict(nan_numeric=[], set_a_rating_cols=None),
    "E2": dict(nan_numeric=["absa_seat_e2", "absa_food_e2", "absa_staff_e2",
                             "absa_ground_service_e2", "absa_entertainment_e2"],
               set_a_rating_cols=None),
}
NATIVE_NAN_MODELS = {"XGBoost", "LightGBM"}


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute Set A rating columns using the group median of a chosen grouping column (default: Type Of Traveller).
    'Unknown' (and any unseen category) always falls back to the overall median, since 06_feature_sets.ipynb found 
    Unknown is too unreliable a group to trust its own median.
    The default group_col="Type Of Traveller" reflects a separate comparison re-validated on train set (07_modelling_pipeline.ipynb, Section 3)
    If that conclusion ever changes, update the group_col argument where this class is called -- not here, since this class
    does not perform that comparison itself.
    All statistics are learned ONLY from fit() data.
    """

    def __init__(self, rating_cols, group_col="Type Of Traveller", unknown_label="Unknown"):
        self.rating_cols = rating_cols
        self.group_col = group_col
        self.unknown_label = unknown_label

    def fit(self, X, y=None):
        known = X[X[self.group_col] != self.unknown_label]
        self.overall_median_ = X[self.rating_cols].median()
        self.group_median_ = known.groupby(self.group_col)[self.rating_cols].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.rating_cols:
            mapped = X[self.group_col].map(self.group_median_[col])
            mapped = mapped.fillna(self.overall_median_[col])
            X[col] = X[col].fillna(mapped)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def clean_numeric_cols_for(df, nan_numeric_cols):
    return [c for c in df.columns
            if c not in CAT_COLS + NON_FEATURE_COLS + nan_numeric_cols
            and pd.api.types.is_numeric_dtype(df[c])]


def split_continuous_and_binary(numeric_cols):
    """
    Separate genuinely continuous columns from 0/1 binary columns. 
    Binary columns are excluded from StandardScaler even in the Logistic Regression branch
    to avoid distorting how L2 regularization weighs it relative to other features.
    """
    binary_cols = [c for c in numeric_cols if c == "Verified" or c.endswith("_missing")]
    continuous_cols = [c for c in numeric_cols if c not in binary_cols]
    return continuous_cols, binary_cols


def get_model(model_name, scale_pos_weight=1.0, n_jobs=-1):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    if model_name == "RandomForest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=n_jobs)
    if model_name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                              n_jobs=n_jobs, scale_pos_weight=scale_pos_weight)
    if model_name == "LightGBM":
        return LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1)
    raise ValueError(model_name)


def build_pipeline(set_name, model_name, train_df, scale_pos_weight=1.0, n_jobs=-1):
    """
    train_df: the Set's training dataframe (used only to read column names/dtypes for ColumnTransformer construction;
    all fitting happens later via pipeline.fit()).
    """
    schema = SET_SCHEMA[set_name]
    nan_numeric = schema["nan_numeric"]
    clean_numeric = clean_numeric_cols_for(train_df, nan_numeric)
    continuous_cols, binary_cols = split_continuous_and_binary(clean_numeric)

    scale_numeric = model_name == "LogisticRegression"
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num_continuous", numeric_transform, continuous_cols),
        ("num_binary", "passthrough", binary_cols),
    ]
    if nan_numeric:
        if model_name in NATIVE_NAN_MODELS:
            transformers.append(("num_nan", "passthrough", nan_numeric))
        else:
            fill_steps = [("fill0", SimpleImputer(strategy="constant", fill_value=0))]
            if scale_numeric:
                fill_steps.append(("scale", StandardScaler()))
            transformers.append(("num_nan", Pipeline(fill_steps), nan_numeric))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    steps = []
    if schema["set_a_rating_cols"] is not None:
        steps.append(("set_a_impute", GroupMedianImputer(schema["set_a_rating_cols"])))
    steps.append(("preprocess", preprocessor))
    steps.append(("model", get_model(model_name, scale_pos_weight, n_jobs=n_jobs)))
    return Pipeline(steps)


def load_sets_and_split(data_dir, set_files, test_frac=0.2):
    """Load all Sets and return (sets, splits) using the shared chronological split."""
    sets = {name: pd.read_csv(data_dir + fname, parse_dates=["Review Date", "Date Flown"])
            for name, fname in set_files.items()}

    sorted_by_date = sets["A"].sort_values(["Review Date", ID_COL], kind="stable").reset_index(drop=True)
    n = len(sorted_by_date)
    cut = int(n * (1 - test_frac))
    train_ids = sorted_by_date.iloc[:cut][ID_COL].values
    test_ids = sorted_by_date.iloc[cut:][ID_COL].values

    splits = {}
    for name, df in sets.items():
        df = df.copy()
        df["Verified"] = df["Verified"].astype(int)
        df_train = df[df[ID_COL].isin(train_ids)].sort_values(["Review Date", ID_COL], kind="stable").reset_index(drop=True)
        df_test = df[df[ID_COL].isin(test_ids)].sort_values(["Review Date", ID_COL], kind="stable").reset_index(drop=True)
        splits[name] = {"train": df_train, "test": df_test}
    return sets, splits