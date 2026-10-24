"""
Data Preparation Layer.

Two responsibilities, mirroring eda.ipynb Sections 13-16:

1. Row-level cleaning (`clean_bookings`) — deterministic, depends only on each row's own
   raw text, never on a statistic computed across rows. Runs ONCE on the full raw dataset,
   before any train/val/test split, so it carries no data-leakage risk.

2. A scikit-learn preprocessing pipeline (`build_preprocessing_pipeline`) — everything that
   involves a *learned* statistic (an outlier cap, an imputed value, a scaling factor, a
   one-hot vocabulary) is fit on the training split only and applied unchanged to
   validation/test. Three custom transformers (`EngineeredFeatureAdder`,
   `MissingnessIndicatorAdder`, `IQRCapper`) are ported directly from the EDA.

All literal cleaning maps, cap columns, and impute strategies are read from config.yaml
rather than hardcoded here, so they can be adjusted without touching this file.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 1. Row-level cleaning (EDA Section 13.1 — clean_bookings())
# =============================================================================
def _canon(s):
    """strip + uppercase + collapse internal whitespace runs."""
    s = s.astype(str).str.strip().str.upper()
    return s.str.replace(r"\s+", " ", regex=True)


def _fix_adr(v):
    """Parse a raw `adr` string: strip an 'SGD' prefix, and disambiguate comma-as-decimal
    from comma-as-thousands-separator (comma with no dot present = decimal separator)."""
    v = v.replace("SGD", "").strip()
    if "," in v and "." not in v:
        v = v.replace(",", ".")
    else:
        v = v.replace(",", "")
    return v


def load_raw_bookings(config: dict) -> pd.DataFrame:
    """Reads the raw `bookings` table straight out of the SQLite database — completely
    unmodified (including the `SGD `-prefixed/comma-decimal `adr` values, `" days"`-suffixed
    `lead_time`, inconsistent casing, typos, and the `Unnamed: 0` junk column). All cleaning
    happens in `clean_bookings()` below; nothing is pre-cleaned at the SQL layer."""
    data_cfg = config["data"]
    con = sqlite3.connect(data_cfg["db_path"])
    try:
        raw = pd.read_sql(f"SELECT * FROM {data_cfg['table']}", con)
    finally:
        con.close()
    return raw


def clean_bookings(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Deterministic structural cleaning: dedup, type casts, categorical canonicalisation,
    target derivation. No row-to-row statistics are used, so this is leakage-safe to run
    before any split. Reproduces eda.ipynb's clean_bookings() exactly, driven by the maps
    in config['cleaning'].

    Parameters
    ----------
    raw : the raw bookings dataframe, exactly as read from data/hotel.db.
    config : the full parsed config.yaml.

    Returns
    -------
    df_model : cleaned dataframe with one row per unique booking, ready for feature
        selection and splitting.
    """
    clean_cfg = config["cleaning"]

    df = raw.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates().copy()

    # --- lead_time: strip " days" suffix, numeric-parse, null out impossible negatives ---
    lt = df["lead_time"].astype(str).str.replace(" days", "", regex=False).str.strip()
    lt = pd.to_numeric(lt, errors="coerce")
    df["lead_time_clean"] = lt.where(lt >= 0)

    # --- adr: strip SGD prefix, disambiguate comma-as-decimal vs. thousands-separator ---
    missing_spellings = set(clean_cfg["missing_value_spellings"])
    df["adr_clean"] = pd.to_numeric(
        df["adr"].astype(str).str.strip().apply(_fix_adr),
        errors="coerce",
    )

    # --- categorical canonicalisation: strip + upper + collapse whitespace ---
    cat_source_cols = [
        "hotel", "meal", "market_segment", "distribution_channel", "deposit_type",
        "customer_type", "reservation_status", "promo_code", "country", "reserved_room_type",
    ]
    for c in cat_source_cols:
        df[c + "_clean"] = _canon(df[c])

    # --- per-column typo fixes (config-driven) ---
    for col, fix_map in clean_cfg["category_typo_fix"].items():
        df[col] = df[col].replace(fix_map)

    # --- consolidate missing-value spellings into each column's own canonical marker ---
    for col, canonical_missing in clean_cfg["missing_value_canonical"].items():
        if col == "country_clean":
            continue  # country has its own ISO-mapping step below
        df[col] = df[col].apply(lambda v, cm=canonical_missing: cm if v in missing_spellings else v)

    # --- country: full name -> ISO code, then fold missing spellings into OTH ---
    df["country_clean"] = df["country_clean"].replace(clean_cfg["country_iso_map"])
    country_missing_marker = clean_cfg["missing_value_canonical"]["country_clean"]
    df["country_clean"] = df["country_clean"].apply(
        lambda v: country_missing_marker if v in missing_spellings else v
    )

    # --- promo_code: blank means "no promo used", semantically distinct from "not recorded" ---
    df["promo_code_clean"] = df["promo_code_clean"].replace({"": "NONE"})

    # --- month: name -> 3-letter code + 1-12 numeric order ---
    month_map = clean_cfg["month_map"]
    month_num_map = clean_cfg["month_num_map"]
    df["month_clean"] = df["arrival_date_month"].astype(str).str.strip().str.upper().map(month_map)
    df["month_num"] = df["month_clean"].map(month_num_map)

    # --- numeric casts for every remaining candidate numeric column ---
    numeric_cast_cols = [
        "arrival_date_year", "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
        "days_in_waiting_list", "required_car_parking_spaces", "total_of_special_requests",
    ]
    for c in numeric_cast_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- assigned_room_type / room_type_mismatch: kept only for reference, NEVER a model
    # feature (only known once a room is allocated, i.e. after booking-time) ---
    df["assigned_room_type_clean"] = _canon(df["assigned_room_type"])
    df["room_type_mismatch"] = (
        df["assigned_room_type_clean"] != df["reserved_room_type_clean"]
    ).astype(int)
    df["has_company"] = (df["company"].astype(str).str.strip() != "").astype(int)
    df["has_agent"] = (df["agent"].astype(str).str.strip() != "").astype(int)

    # --- targets ---
    df["outcome"] = df["reservation_status_clean"]
    df["canceled_flag"] = (df["outcome"] == "CANCELED").astype(int)
    df["noshow_flag"] = (df["outcome"] == "NO-SHOW").astype(int)
    df["checkout_flag"] = (df["outcome"] == "CHECK-OUT").astype(int)

    return df


# =============================================================================
# 2. Custom sklearn-compatible transformers (EDA Sections 13.3 / 14 / 16.1)
# =============================================================================
class MissingnessIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adds a `<col>_was_missing` binary flag for every column (from `columns`, or
    auto-detected at fit time) that has missing values in the *training* data — learned
    once on train, applied identically to validation/test so the flag set never depends
    on what happens to be missing in the evaluation split."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        self.cols_with_na_ = [c for c in cols if X[c].isna().any()]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_with_na_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X


class EngineeredFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds the 11 derived features ranked in EDA Section 10 (has_agent/has_company are
    already present after clean_bookings(), so they don't need to be re-added here).
    `year_trend` needs a genuine fit step (relative to the minimum arrival year seen in
    training data) — every other feature is purely row-wise."""

    def fit(self, X, y=None):
        self.min_year_ = X["arrival_date_year"].min()
        return self

    def transform(self, X):
        X = X.copy()
        X["is_domestic"] = (X["country_clean"] == "SGP").astype(int)
        X["prior_cancellation_flag"] = (X["previous_cancellations"] > 0).astype(int)
        X["total_guests"] = X["adults"].fillna(0) + X["children"].fillna(0) + X["babies"].fillna(0)
        X["has_children"] = ((X["children"].fillna(0) + X["babies"].fillna(0)) > 0).astype(int)
        X["total_nights"] = X["stays_in_weekend_nights"].fillna(0) + X["stays_in_week_nights"].fillna(0)
        X["commitment_signal"] = (
            X["total_of_special_requests"].fillna(0) + X["required_car_parking_spaces"].fillna(0)
        )
        X["high_risk_combo"] = (
            (X["deposit_type_clean"] == "NON REFUND")
            & (X["market_segment_clean"].isin(["GROUPS", "OFFLINE TA/TO"]))
        ).astype(int)
        X["lead_time_log"] = np.log1p(X["lead_time_clean"])
        X["month_sin"] = np.sin(2 * np.pi * X["month_num"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month_num"] / 12)
        X["year_trend"] = X["arrival_date_year"] - self.min_year_
        return X


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - k*IQR, Q3 + k*IQR], with Q1/Q3/IQR learned from fit()
    (training data only). k=3.0 (vs. the conventional 1.5) is deliberately wide — EDA
    Section 8 showed these "outliers" carry real signal, so this caps only the most
    extreme leverage points rather than the same broad swath Section 8 characterised as
    informative."""

    def __init__(self, columns, k=3.0):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in self.columns:
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lower=lo, upper=hi)
        return X


# =============================================================================
# 3. Full preprocessing pipeline (EDA Section 16.2)
# =============================================================================
def build_preprocessing_pipeline(numeric_cols, categorical_cols, config: dict) -> Pipeline:
    """feature engineering -> missingness flags -> impute/cap/scale/encode, in one
    scikit-learn Pipeline. Fit only on the training split; transform-only on val/test."""
    prep_cfg = config["preprocessing"]
    cap_cols = prep_cfg["outlier_cap"]["columns"]
    cap_k = prep_cfg["outlier_cap"]["k"]
    missing_cols = prep_cfg["missingness_indicator_columns"]

    numeric_pipe = Pipeline([
        ("cap", IQRCapper(columns=cap_cols, k=cap_k)),
        ("impute", SimpleImputer(strategy=prep_cfg["numeric_impute_strategy"])),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=prep_cfg["categorical_impute_strategy"])),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    coltrans = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return Pipeline([
        ("feat_eng", EngineeredFeatureAdder()),
        ("miss_ind", MissingnessIndicatorAdder(columns=missing_cols)),
        ("coltrans", coltrans),
    ])


# =============================================================================
# 4. Top-level entry point
# =============================================================================
def prepare_data(config: dict) -> pd.DataFrame:
    """Loads the raw bookings table from data/hotel.db and applies row-level cleaning.
    Returns the cleaned dataframe (`df_model`), from which both tasks' feature/target
    columns are selected downstream in model_training.py."""
    raw = load_raw_bookings(config)
    df_model = clean_bookings(raw, config)
    return df_model
