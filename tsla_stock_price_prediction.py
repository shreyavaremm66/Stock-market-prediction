#!/usr/bin/env python3
# filepath: tsla_stock_price_prediction.py
"""
Tesla (TSLA) Stock Price Prediction – End-to-End Script

Why this layout:
- Keeps a clear separation between data prep, modeling, and evaluation for maintainability.
- Time-series safe: no shuffling, chronological splits, and rolling CV.
- Multiple baselines ensure the ML model is meaningfully better.

Usage:
    python tsla_stock_price_prediction.py --csv /mnt/data/TSLA.csv --horizon 1

Outputs:
- Metrics for train/valid/test
- Matplotlib plots (shown if --no-plot not set)
- Saved model at /mnt/data/tsla_model.joblib and features at /mnt/data/tsla_feature_names.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import joblib  # preferred
except Exception:  # pragma: no cover
    joblib = None
    import pickle


# -----------------------------
# Data Loading & Preparation
# -----------------------------

def _infer_date_column(cols: List[str]) -> str:
    """Infer likely date column name.
    Why: User CSVs vary; robustly pick the date column to avoid manual fixes.
    """
    candidates = [c for c in cols if c.lower() in {"date", "timestamp", "datetime"}]
    if candidates:
        return candidates[0]
    # Fallback: choose the first column
    return cols[0]


def load_tsla_csv(path: str) -> pd.DataFrame:
    """Load CSV, parse date, and enforce sorted ascending by date."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty.")

    date_col = _infer_date_column(df.columns.tolist())
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    # Standardize price column name to Close; prefer Adj Close if present.
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"].astype(float)
    elif "Close" in df.columns:
        df["Close"] = df["Close"].astype(float)
    else:
        raise ValueError("CSV must contain either 'Adj Close' or 'Close'.")

    # Ensure required columns exist; fill absent ones safely.
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            # Minimal fallback to keep pipeline running; forecast quality may degrade.
            df[col] = np.nan

    # Coerce numeric types
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


# -----------------------------
# Feature Engineering
# -----------------------------

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use Wilder's smoothing via ewm with adjust=False
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags, rolling stats, and technical indicators.
    Why: Enrich simple price data with momentum/volatility structure to aid learning.
    """
    out = df.copy()

    # Basic ranges
    out["HL_Range"] = out["High"] - out["Low"]
    out["CO_Range"] = out["Close"] - out["Open"]

    # Returns
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = np.log1p(out["Return"])  # stable for small values

    # Lags
    for l in range(1, 11):
        out[f"Close_lag{l}"] = out["Close"].shift(l)
    for l in range(1, 6):
        out[f"Volume_lag{l}"] = out["Volume"].shift(l)

    # Rolling stats
    for w in (5, 10, 20):
        out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
        out[f"Volatility_{w}"] = out["Return"].rolling(w).std()

    # EMA for momentum
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    macd, sig = _macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd
    out["MACD_signal"] = sig
    out["MACD_hist"] = macd - sig

    # Bollinger Bands (20)
    m = out["Close"].rolling(20).mean()
    s = out["Close"].rolling(20).std()
    out["BB_mid_20"] = m
    out["BB_up_20"] = m + 2 * s
    out["BB_dn_20"] = m - 2 * s
    out["BB_pctB_20"] = (out["Close"] - out["BB_dn_20"]) / ((out["BB_up_20"] - out["BB_dn_20"]) + 1e-12)

    # RSI
    out["RSI_14"] = _rsi(out["Close"], 14)

    # Day-of-week cyclic encodings
    out["DOW"] = out["Date"].dt.dayofweek
    out["DOW_sin"] = np.sin(2 * np.pi * out["DOW"] / 7)
    out["DOW_cos"] = np.cos(2 * np.pi * out["DOW"] / 7)

    return out


def make_supervised(df: pd.DataFrame, target_col: str = "Close", horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """Create next-step target by shifting negative (predict t+h from t)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    y = df[target_col].shift(-horizon)
    X = df.drop(columns=[c for c in df.columns if c in {"Date"}])  # keep all engineered features
    # Align
    X = X.iloc[:-horizon, :]
    y = y.iloc[:-horizon]
    # Remove rows with NaNs from feature creation
    valid = X.dropna().index
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y


# -----------------------------
# Splitting & Baselines
# -----------------------------

def chronological_split(X: pd.DataFrame, y: pd.Series, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple:
    """Time-ordered split without leakage."""
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.0")
    n = len(X)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_train + n_valid], y.iloc[n_train:n_train + n_valid]
    X_test, y_test = X.iloc[n_train + n_valid:], y.iloc[n_train + n_valid:]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def baseline_predictions(y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series, X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Produce naive and SMA baselines across splits.
    Why: Ensures ML outperforms simple heuristics.
    """
    baselines = {}

    # Naive: predict previous day's close
    def _shift_last(close_series: pd.Series) -> np.ndarray:
        return close_series.shift(1).iloc[1:].to_numpy()

    # Align naive for each split
    for split_name, Xs, ys, Xt, yt in [
        ("valid", pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), X_valid, y_valid),
        ("test", pd.concat([X_train, X_valid, X_test]), pd.concat([y_train, y_valid, y_test]), X_test, y_test),
    ]:
        closes = Xs["Close"]
        pred_naive = closes.shift(1).iloc[-len(yt):].to_numpy()

        # SMA(5)
        sma5 = closes.rolling(5).mean()
        pred_sma5 = sma5.shift(1).iloc[-len(yt):].to_numpy()

        baselines[split_name] = {
            "naive": pred_naive,
            "sma5": pred_sma5,
        }

    return baselines


# -----------------------------
# Modeling
# -----------------------------

def _time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def build_model_grids(random_state: int = 42) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    """Define models and small, safe hyperparameter grids for quick tuning."""
    models: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}

    # Ridge (linear) – benefits from scaling
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge())
    ])
    ridge_grid = {"model__alpha": [0.1, 1.0, 10.0, 50.0]}
    models["ridge"] = (ridge_pipe, ridge_grid)

    # Gradient Boosting – robust on tabular time-series features
    gbr_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", GradientBoostingRegressor(random_state=random_state))
    ])
    gbr_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [1.0]
    }
    models["gbr"] = (gbr_pipe, gbr_grid)

    # Random Forest – strong baseline; scaling not required but harmless
    rf_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))
    ])
    rf_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [6, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    models["rf"] = (rf_pipe, rf_grid)

    return models


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: Dict
    best_score_rmse: float


def train_with_rolling_cv(X: pd.DataFrame, y: pd.Series, model_name: str, pipe: Pipeline, grid: Dict[str, List], n_splits: int = 5) -> FitResult:
    cv = _time_series_cv(n_splits=n_splits)
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    gscv.fit(X, y)
    return FitResult(
        name=model_name,
        best_estimator=gscv.best_estimator_,
        best_params=gscv.best_params_,
        best_score_rmse=-gscv.best_score_,
    )


# -----------------------------
# Evaluation & Plotting
# -----------------------------

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}


def print_report(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        print(f"{k:>7}: {v:,.4f}")


def plot_actual_vs_pred(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure()
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Persistence
# -----------------------------

def save_model(model: Pipeline, feature_names: List[str], model_path: str = "/mnt/data/tsla_model.joblib", feat_path: str = "/mnt/data/tsla_feature_names.json") -> None:
    if joblib is not None:
        joblib.dump(model, model_path)
    else:  # pragma: no cover
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved model to: {model_path}")
    print(f"Saved feature names to: {feat_path}")


# -----------------------------
# Inference helper
# -----------------------------

def predict_next_day(latest_df: pd.DataFrame, model: Pipeline, horizon: int = 1) -> float:
    """Predict next-day close based on the last available feature row."""
    if latest_df.empty:
        raise ValueError("latest_df is empty.")
    last_row = latest_df.drop(columns=["Date"]).iloc[[-horizon]]  # use most recent features
    pred = float(model.predict(last_row)[0])
    return pred


# -----------------------------
# Main CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to TSLA CSV.")
    parser.add_argument("--horizon", type=int, default=1, help="Days ahead to predict (default: 1).")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # 1) Load & feature engineering
    raw = load_tsla_csv(args.csv)
    print(f"Loaded {len(raw):,} rows; date range: {raw['Date'].min().date()} -> {raw['Date'].max().date()}")

    feat = add_technical_features(raw)
    X, y = make_supervised(feat, target_col="Close", horizon=args.horizon)

    # 2) Chronological split
    X_train, y_train, X_valid, y_valid, X_test, y_test = chronological_split(X, y, (0.70, 0.15, 0.15))
    print(f"Train/Valid/Test sizes: {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}")

    # 3) Baselines
    baselines = baseline_predictions(y_train, y_valid, y_test, X_train, X_valid, X_test)
    # Validation
    yv = y_valid.to_numpy()
    bv_naive = baselines["valid"]["naive"]
    bv_sma5 = baselines["valid"]["sma5"]
    print_report("Baseline – Validation (Naive)", regression_report(yv, bv_naive))
    print_report("Baseline – Validation (SMA5)", regression_report(yv, bv_sma5))

    # 4) Models + CV on Train set only to prevent peeking; then evaluate on Valid
    models = build_model_grids(random_state=args.seed)

    best_overall: Tuple[str, FitResult, np.ndarray] | None = None
    for name, (pipe, grid) in models.items():
        print(f"\nTuning {name}...")
        fit_res = train_with_rolling_cv(X_train, y_train, name, pipe, grid, n_splits=5)
        # Fit on train, evaluate on valid
        model = fit_res.best_estimator
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics_valid = regression_report(y_valid, pred_valid)
        print_report(f"{name.upper()} – Validation", metrics_valid)
        print(f"Best params: {fit_res.best_params}")

        rmse = metrics_valid["RMSE"]
        if (best_overall is None) or (rmse < best_overall[1].best_score_rmse):
            # Note: using actual valid RMSE for comparison
            best_overall = (name, fit_res, pred_valid)

    assert best_overall is not None
    best_name, best_fitres, _ = best_overall
    print(f"\nSelected model: {best_name} (CV RMSE: {best_fitres.best_score_rmse:.4f})")

    # 5) Refit on Train+Valid, evaluate on Test
    best_model = best_fitres.best_estimator
    X_trv = pd.concat([X_train, X_valid])
    y_trv = pd.concat([y_train, y_valid])
    best_model.fit(X_trv, y_trv)

    y_pred_test = best_model.predict(X_test)
    metrics_test = regression_report(y_test, y_pred_test)
    print_report("BEST MODEL – Test", metrics_test)

    # 6) Plot test predictions
    if not args.no_plot:
        test_dates = raw.loc[X_test.index + args.horizon, "Date"]  # align to target dates
        plot_actual_vs_pred(test_dates, y_test.to_numpy(), y_pred_test, f"TSLA – {best_name.upper()} Test Predictions")

    # 7) Save model & features
    save_model(best_model, feature_names=X.columns.tolist())

    # 8) Next-day prediction from the very latest features
    try:
        next_pred = predict_next_day(feat, best_model, horizon=1)
        print(f"\nNext-day predicted Close: {next_pred:,.2f}")
    except Exception as e:
        print(f"Skipping next-day prediction: {e}")


if __name__ == "__main__":
    # Fail fast with helpful message
    try:
        main()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("CSV appears to be empty or invalid.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
