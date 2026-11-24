"""Daily and aggregate store-level sales analytics.

This module receives normalized orders from Stage 1. It does not read the
original WooCommerce export and does not apply a second status policy.
"""
from __future__ import annotations

import pandas as pd


def daily_sales_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """Return one row per calendar day, including zero-sales days.

    Expected input is the cleaned orders frame produced by csv_reader.py.
    The CSV reader currently determines eligible order statuses.
    """
    columns = ["date", "revenue", "orders"]
    if orders is None or orders.empty:
        return pd.DataFrame(columns=columns)

    required = {"order_date", "order_total"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(
            "Cannot aggregate daily sales; missing columns: "
            + ", ".join(sorted(missing))
        )

    data = orders[["order_date", "order_total"]].copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["order_total"] = pd.to_numeric(data["order_total"], errors="coerce")
    data = data.dropna(subset=["order_date", "order_total"])

    if data.empty:
        return pd.DataFrame(columns=columns)

    data["date"] = data["order_date"].dt.floor("D")
    daily = (
        data.groupby("date", sort=True)
        .agg(
            revenue=("order_total", "sum"),
            orders=("order_total", "size"),
        )
        .sort_index()
    )

    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0)
    daily.index.name = "date"
    daily["revenue"] = daily["revenue"].astype(float)
    daily["orders"] = daily["orders"].astype(int)
    return daily.reset_index()[columns]


def build_sales_analysis(orders: pd.DataFrame, items: pd.DataFrame) -> dict:
    """Build dashboard KPIs and chart series from cleaned order/item frames."""
    daily = daily_sales_frame(orders)

    revenue = (
        float(pd.to_numeric(orders["order_total"], errors="coerce").fillna(0).sum())
        if orders is not None and not orders.empty and "order_total" in orders
        else 0.0
    )
    order_count = (
        int(orders["order_id"].nunique())
        if orders is not None and not orders.empty and "order_id" in orders
        else int(len(orders)) if orders is not None else 0
    )

    if not daily.empty:
        monthly_frame = daily.assign(month=daily["date"].dt.to_period("M"))
        monthly = monthly_frame.groupby("month", sort=True)["revenue"].sum()
        monthly_labels = [period.strftime("%b %Y") for period in monthly.index]
        monthly_values = [round(float(value), 2) for value in monthly.values]
        recent_30 = float(daily["revenue"].tail(30).sum())
        previous_30 = (
            float(daily["revenue"].iloc[-60:-30].sum())
            if len(daily) >= 60 else 0.0
        )
        growth = (
            (recent_30 - previous_30) / abs(previous_30) * 100
            if previous_30 != 0 else None
        )
        date_start = daily["date"].min().strftime("%Y-%m-%d")
        date_end = daily["date"].max().strftime("%Y-%m-%d")
    else:
        monthly_labels, monthly_values = [], []
        growth, date_start, date_end = None, None, None

    units = (
        float(pd.to_numeric(items["quantity_individual_units"], errors="coerce").fillna(0).sum())
        if items is not None and not items.empty and "quantity_individual_units" in items
        else float(pd.to_numeric(items["quantity"], errors="coerce").fillna(0).sum())
        if items is not None and not items.empty and "quantity" in items
        else 0.0
    )

    return {
        "revenue": round(revenue, 2),
        "orders": order_count,
        "aov": revenue / order_count if order_count else 0.0,
        "units": units,
        "daily_labels": daily["date"].dt.strftime("%Y-%m-%d").tolist() if not daily.empty else [],
        "daily_values": [round(float(value), 2) for value in daily["revenue"]] if not daily.empty else [],
        "monthly_labels": monthly_labels,
        "monthly_values": monthly_values,
        "growth": growth,
        "date_start": date_start,
        "date_end": date_end,
        "raw_orders": int(len(orders)) if orders is not None else 0,
    }
