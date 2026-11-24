"""Store-level WooCommerce daily revenue forecasting.

Features:
- Leakage-safe lag, rolling, volatility, median, EWM and calendar features.
- Expanding-window (rolling-origin) backtesting.
- Recursive 7/30/60/90-day evaluation and future forecasts.
- Seasonal Naive, Random Forest, HistGradientBoosting and XGBoost.
- Residual diagnostics and permutation feature importance.

Input compatibility:
    run_forecasting(orders, horizon, artifact_dir)

`orders` must contain order_date and order_total. It should already be the
cleaned/eligible order frame produced by the CSV pipeline. This module does
not decide which WooCommerce order statuses are financially eligible.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLLING_WINDOWS = (7, 14, 30)
EVALUATION_HORIZONS = (7, 30, 60, 90)
N_SPLITS = 4
MIN_TRAIN_DAYS = 90


def _wape(actual, predicted):
    """Weighted absolute percentage error; returns NaN for zero denominator."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator * 100.0)


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": _wape(actual, predicted),
        "bias": float(np.mean(predicted - actual)),
    }


def _feature_row(history: pd.Series, next_date: pd.Timestamp) -> dict:
    """Build one feature row using only observations available before next_date."""
    values = history.astype(float)
    row = {
        "dow": int(next_date.dayofweek),
        "day": int(next_date.day),
        "month": int(next_date.month),
        "week": int(next_date.isocalendar().week),
        "quarter": int(next_date.quarter),
        "weekend": int(next_date.dayofweek >= 5),
    }

    for lag in LAGS:
        row[f"lag_{lag}"] = float(values.iloc[-lag]) if len(values) >= lag else 0.0

    shifted = values
    for window in ROLLING_WINDOWS:
        sample = shifted.tail(window)
        row[f"roll_{window}_mean"] = float(sample.mean()) if len(sample) else 0.0
        row[f"roll_{window}_median"] = float(sample.median()) if len(sample) else 0.0
        row[f"roll_{window}_std"] = float(sample.std(ddof=0)) if len(sample) else 0.0

    row["ewm_7"] = float(values.ewm(span=7, adjust=False).mean().iloc[-1]) if len(values) else 0.0
    row["ewm_14"] = float(values.ewm(span=14, adjust=False).mean().iloc[-1]) if len(values) else 0.0
    row["same_weekday_avg_4w"] = (
        float(values.iloc[-7:: -7].head(4).mean())
        if len(values) >= 7 else float(values.mean()) if len(values) else 0.0
    )
    return row


def _feature_frame(history: pd.Series) -> pd.DataFrame:
    """Create supervised features; every row only uses prior-day observations."""
    rows = []
    for position, date in enumerate(history.index):
        if position == 0:
            rows.append({})
            continue
        prior = history.iloc[:position]
        rows.append(_feature_row(prior, pd.Timestamp(date)))
    features = pd.DataFrame(rows, index=history.index)
    return features.replace([np.inf, -np.inf], np.nan)


def _new_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=2.0,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.2,
            reg_lambda=3.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _recursive_forecast(model, history: pd.Series, horizon: int) -> list[float]:
    """Forecast recursively: predictions, never future actuals, feed later steps."""
    history = history.copy().astype(float).sort_index()
    predictions = []

    for _ in range(int(horizon)):
        next_date = pd.Timestamp(history.index[-1]) + pd.Timedelta(days=1)
        row = _feature_row(history, next_date)
        X_next = pd.DataFrame([row])
        prediction = float(model.predict(X_next)[0])
        prediction = max(0.0, prediction) if np.isfinite(prediction) else 0.0
        predictions.append(prediction)
        history.loc[next_date] = prediction

    return predictions


def _seasonal_recursive(history: pd.Series, horizon: int) -> list[float]:
    """Repeat the last observed seven-day pattern recursively."""
    values = history.astype(float).tolist()
    predictions = []
    for _ in range(int(horizon)):
        prediction = max(0.0, float(values[-7])) if len(values) >= 7 else max(0.0, float(np.mean(values)))
        predictions.append(prediction)
        values.append(prediction)
    return predictions


def _make_supervised(history: pd.Series):
    features = _feature_frame(history)
    frame = features.join(history.rename("target")).replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def _origin_positions(n_rows: int, horizon: int, min_train: int, n_splits: int = N_SPLITS):
    """Return expanding-window origins with non-overlapping validation blocks."""
    available = n_rows - horizon - min_train
    if available < 0:
        return []
    # Spread origins across the available history; keep at most n_splits folds.
    count = min(n_splits, max(1, available // max(7, horizon // 2) + 1))
    if count == 1:
        return [n_rows - horizon]
    starts = np.linspace(min_train, n_rows - horizon, num=count, dtype=int)
    return sorted(set(int(x) for x in starts if x >= min_train and x + horizon <= n_rows))


def _evaluate_horizon(daily: pd.Series, horizon: int):
    """Evaluate recursive predictions over expanding-window origins."""
    n = len(daily)
    origins = _origin_positions(n, horizon, MIN_TRAIN_DAYS)
    if not origins:
        return None

    predictions_by_model = {"Seasonal Naive": []}
    actual_all = []
    dates_all = []
    fold_details = []

    for model_name in _new_models():
        predictions_by_model[model_name] = []

    for fold_number, origin in enumerate(origins, start=1):
        train_history = daily.iloc[:origin].copy()
        actual = daily.iloc[origin:origin + horizon].to_numpy(dtype=float)
        future_dates = daily.index[origin:origin + horizon]

        if len(actual) != horizon:
            continue

        seasonal = _seasonal_recursive(train_history, horizon)
        predictions_by_model["Seasonal Naive"].extend(seasonal)

        supervised = _make_supervised(train_history)
        if supervised.empty or len(supervised) < 30:
            continue

        X_train = supervised.drop(columns=["target"])
        y_train = supervised["target"].astype(float)

        for model_name, estimator in _new_models().items():
            # A fresh estimator is fitted independently at each forecast origin.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, y_train)
            pred = _recursive_forecast(estimator, train_history, horizon)
            predictions_by_model[model_name].extend(pred)

        actual_all.extend(actual.tolist())
        dates_all.extend(future_dates.tolist())
        fold_details.append({
            "fold": fold_number,
            "train_size": int(origin),
            "validation_size": int(horizon),
            "start": future_dates[0].strftime("%Y-%m-%d"),
            "end": future_dates[-1].strftime("%Y-%m-%d"),
        })

    if not actual_all:
        return None

    actual_array = np.asarray(actual_all, dtype=float)
    result = {
        "horizon": int(horizon),
        "folds": fold_details,
        "fold_count": len(fold_details),
        "actual": actual_array,
        "dates": pd.to_datetime(dates_all),
        "predictions": {},
        "metrics": [],
    }

    for model_name, values in predictions_by_model.items():
        pred_array = np.asarray(values, dtype=float)
        if len(pred_array) != len(actual_array):
            continue
        result["predictions"][model_name] = pred_array
        score = _metrics(actual_array, pred_array)
        result["metrics"].append({
            "model": model_name,
            **score,
        })

    result["metrics"].sort(key=lambda row: row["wape"] if np.isfinite(row["wape"]) else float("inf"))
    return result


def _prepare_daily(orders: pd.DataFrame):
    required = {"order_date", "order_total"}
    if orders is None or not required.issubset(orders.columns):
        raise ValueError("The cleaned orders data must contain order_date and order_total.")

    data = orders[["order_date", "order_total"]].copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["order_total"] = pd.to_numeric(data["order_total"], errors="coerce")
    data = data.dropna(subset=["order_date", "order_total"])
    if data.empty:
        raise ValueError("No valid order dates and order totals were found.")

    data["date"] = data["order_date"].dt.floor("D")
    daily = data.groupby("date")["order_total"].sum().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(calendar, fill_value=0.0).astype(float)


def _feature_importance(model, X, y, feature_names):
    """Permutation importance on a chronological trailing validation segment."""
    if len(X) < 10:
        return []
    split = max(1, int(len(X) * 0.8))
    X_eval, y_eval = X.iloc[split:], y.iloc[split:]
    if len(X_eval) < 5:
        X_eval, y_eval = X, y

    try:
        result = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        rows = [
            {
                "feature": feature_names[i],
                "importance": round(float(result.importances_mean[i]), 4),
            }
            for i in range(len(feature_names))
        ]
        rows.sort(key=lambda row: row["importance"], reverse=True)
        return rows[:15]
    except Exception:
        return []


def _unavailable(message):
    return {
        "available": False,
        "message": message,
        "metrics": [],
        "labels": [],
        "values": [],
        "actual_labels": [],
        "actual_values": [],
        "predicted_values": [],
        "backtest_rows": [],
        "horizon_metrics": [],
        "feature_importance": [],
        "residual_summary": {},
        "residual_rows": [],
        "folds": [],
    }


def run_forecasting(orders, horizon, artifact_dir):
    """Run rolling-origin model evaluation and generate a recursive forecast."""
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        horizon = 30
    if horizon not in EVALUATION_HORIZONS:
        horizon = 30

    try:
        daily = _prepare_daily(orders)
    except ValueError as exc:
        return _unavailable(str(exc))

    if len(daily) < MIN_TRAIN_DAYS + 7:
        return _unavailable(
            f"At least {MIN_TRAIN_DAYS + 7} calendar days are recommended; "
            f"the cleaned dataset contains {len(daily)}."
        )

    # Run recursive rolling-origin evaluation for all supported horizons.
    horizon_results = {}
    for evaluation_horizon in EVALUATION_HORIZONS:
        evaluation = _evaluate_horizon(daily, evaluation_horizon)
        if evaluation is not None:
            horizon_results[evaluation_horizon] = evaluation

    selected = horizon_results.get(horizon)
    if selected is None:
        # Fall back to the largest evaluable horizon, rather than returning
        # misleading metrics for an evaluation that could not be performed.
        available_horizons = sorted(horizon_results)
        if not available_horizons:
            return _unavailable(
                "Not enough history for rolling-origin evaluation. "
                "Use more historical data or a shorter forecast horizon."
            )
        horizon = min(available_horizons, key=lambda h: abs(h - horizon))
        selected = horizon_results[horizon]

    finite = [
        row for row in selected["metrics"]
        if np.isfinite(row["wape"])
    ]
    winner = finite[0]["model"] if finite else "Seasonal Naive"
    chosen_predictions = selected["predictions"].get(
        winner, selected["predictions"]["Seasonal Naive"]
    )

    # Detailed recursive out-of-fold predictions.
    backtest_rows = []
    residual_values = []
    for date, actual, predicted in zip(
        selected["dates"], selected["actual"], chosen_predictions
    ):
        error = float(actual - predicted)
        residual_values.append(error)
        pct = abs(error) / abs(float(actual)) * 100.0 if actual != 0 else None
        backtest_rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "actual_revenue": round(float(actual), 2),
            "predicted_revenue": round(float(predicted), 2),
            "error": round(error, 2),
            "absolute_error": round(abs(error), 2),
            "error_pct": round(pct, 2) if pct is not None else None,
        })

    residual_array = np.asarray(residual_values, dtype=float)
    residual_summary = {
        "mean_error": round(float(np.mean(residual_array)), 2),
        "mean_absolute_error": round(float(np.mean(np.abs(residual_array))), 2),
        "residual_std": round(float(np.std(residual_array)), 2),
        "overprediction_days": int(np.sum(residual_array < 0)),
        "underprediction_days": int(np.sum(residual_array > 0)),
        "zero_error_days": int(np.sum(residual_array == 0)),
    }
    residual_rows = [
        {
            "date": row["date"],
            "residual": row["error"],
            "absolute_residual": row["absolute_error"],
        }
        for row in backtest_rows
    ]

    # Train final selected model on all available historical supervised rows.
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    final_model = None
    importance_rows = []

    if winner == "Seasonal Naive":
        future_values = _seasonal_recursive(daily, horizon)
        model_path = None
    else:
        final_supervised = _make_supervised(daily)
        X_final = final_supervised.drop(columns=["target"])
        y_final = final_supervised["target"].astype(float)
        final_model = _new_models()[winner]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(X_final, y_final)
        joblib.dump(final_model, artifact_path / "best_sales_model.joblib")
        model_path = str(artifact_path / "best_sales_model.joblib")
        importance_rows = _feature_importance(
            final_model, X_final, y_final, list(X_final.columns)
        )
        future_values = _recursive_forecast(final_model, daily, horizon)

    future_dates = pd.date_range(
        daily.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    formatted_metrics = []
    for row in selected["metrics"]:
        formatted_metrics.append({
            "model": row["model"],
            "mae": round(float(row["mae"]), 2),
            "rmse": round(float(row["rmse"]), 2),
            "wape": round(float(row["wape"]), 2) if np.isfinite(row["wape"]) else None,
            "bias": round(float(row["bias"]), 2),
        })

    horizon_metrics = []
    for h, evaluation in sorted(horizon_results.items()):
        for row in evaluation["metrics"]:
            horizon_metrics.append({
                "horizon": int(h),
                "model": row["model"],
                "mae": round(float(row["mae"]), 2),
                "rmse": round(float(row["rmse"]), 2),
                "wape": round(float(row["wape"]), 2) if np.isfinite(row["wape"]) else None,
                "bias": round(float(row["bias"]), 2),
                "fold_count": int(evaluation["fold_count"]),
            })

    return {
        "available": True,
        "metrics": formatted_metrics,
        "winner": winner,
        "labels": [date.strftime("%Y-%m-%d") for date in future_dates],
        "values": [round(float(value), 2) for value in future_values],
        "horizon": int(horizon),
        "test_size": int(len(selected["actual"])),
        "train_size": int(selected["folds"][-1]["train_size"]) if selected["folds"] else 0,
        "actual_labels": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in selected["dates"]],
        "actual_values": [round(float(value), 2) for value in selected["actual"]],
        "predicted_values": [round(float(value), 2) for value in chosen_predictions],
        "backtest_rows": backtest_rows,
        "horizon_metrics": horizon_metrics,
        "folds": selected["folds"],
        "feature_importance": importance_rows,
        "residual_summary": residual_summary,
        "residual_rows": residual_rows,
        "model_path": model_path,
        "evaluation_method": "expanding-window recursive rolling-origin backtesting",
    }
