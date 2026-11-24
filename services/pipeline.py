"""File-based processing stages for Commerce Intelligence AI.

Stage 1: clean and validate the WooCommerce CSV.
Stage 2: create reusable analytics datasets from Stage 1 artifacts.
Stage 3: forecasting consumes the daily_sales.csv artifact.

No database, WooCommerce API, or external API is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from services.csv_reader import load_orders_csv
from services.sales_analysis import daily_sales_frame, build_sales_analysis
from services.product_analysis import build_product_analysis
from services.customer_analysis import build_customer_analysis
from services.discount_analysis import build_discount_analysis


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _safe_report(data: dict) -> dict:
    """Keep diagnostic output aggregate-only; never store raw customer details."""
    numeric = data.get("numeric_diagnostics", {})
    return {
        "raw_rows": int(data.get("raw_rows", 0)),
        "clean_order_rows": int(len(data["orders"])),
        "product_line_rows": int(len(data["items"])),
        "blank_order_ids": int(data.get("blank_order_ids", 0)),
        "duplicate_orders": int(data.get("duplicate_orders", 0)),
        "blank_required": data.get("blank_required", {}),
        "missing_dates": int(data.get("missing_dates", 0)),
        "product_slots_detected": int(data.get("product_slots_detected", 0)),
        "missing_sku_items": int(data.get("missing_sku_items", 0)),
        "numeric_diagnostics": numeric,
        "warnings": [
            f"{name}: {values['invalid_nonblank']} invalid non-empty value(s)"
            for name, values in numeric.items()
            if values.get("invalid_nonblank", 0)
        ],
    }


def clean_and_validate(source_csv: str | Path, output_dir: str | Path) -> dict:
    """Stage 1: parse source once and persist normalized, PII-minimized datasets."""
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_orders_csv(source_csv)
    orders = data["orders"].copy()
    items = data["items"].copy()

    if orders.empty:
        raise ValueError("No usable orders found after cleaning.")

    # Persist normalized data only. Contact details, addresses, IPs and notes
    # are excluded by csv_reader.py before this point.
    orders.to_csv(output_dir / "cleaned_orders.csv", index=False)
    items.to_csv(output_dir / "normalized_product_lines.csv", index=False)

    report = _safe_report(data)
    report["stage"] = "cleaned"
    report["source_filename"] = source_csv.name
    _write_json(output_dir / "cleaning_report.json", report)
    return report


def load_cleaned_data(output_dir: str | Path) -> dict:
    """Load normalized Stage 1 artifacts; never re-read the original upload."""
    output_dir = Path(output_dir)
    orders_path = output_dir / "cleaned_orders.csv"
    items_path = output_dir / "normalized_product_lines.csv"

    if not orders_path.is_file() or not items_path.is_file():
        raise FileNotFoundError("Cleaned data is not available. Run Stage 1 first.")

    orders = pd.read_csv(orders_path, keep_default_na=False)
    items = pd.read_csv(items_path, keep_default_na=False)

    for frame in (orders, items):
        for column in ("order_date", "paid_date", "date_completed"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

    numeric_groups = (
        (
            orders,
            [
                "order_total", "discount_total", "order_subtotal", "shipping_total",
                "fee_total", "tax_total", "cart_discount", "order_discount",
                "discount_tax",
            ],
        ),
        (
            items,
            [
                "quantity", "quantity_individual_units", "line_total",
                "line_subtotal",
            ],
        ),
    )
    for frame, columns in numeric_groups:
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    return {"orders": orders, "items": items}


def generate_analytics(output_dir: str | Path) -> dict:
    """Stage 2: create dashboard and forecasting datasets from cleaned files."""
    output_dir = Path(output_dir)
    data = load_cleaned_data(output_dir)
    orders, items = data["orders"], data["items"]

    # All downstream reports use the same cleaned order/item data.
    daily = daily_sales_frame(orders)
    sales = build_sales_analysis(orders, items)
    products = build_product_analysis(items)
    customers = build_customer_analysis(orders)
    discounts = build_discount_analysis(orders)

    # This is the canonical daily revenue dataset for the forecasting stage.
    daily.to_csv(output_dir / "daily_sales.csv", index=False)
    products["table"].to_csv(output_dir / "product_performance.csv", index=False)

    customer_table = customers.get("table")
    if customers.get("available") and isinstance(customer_table, pd.DataFrame):
        customer_table.to_csv(output_dir / "customer_rfm.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "customer_rfm.csv", index=False)

    summary = {
        "stage": "analytics_ready",
        "orders": int(orders["order_id"].nunique()) if "order_id" in orders else len(orders),
        "revenue": round(float(orders["order_total"].sum()), 2) if "order_total" in orders else 0.0,
        "units": round(float(items["quantity_individual_units"].sum()), 2)
                 if not items.empty and "quantity_individual_units" in items else 0.0,
        "products": int(products.get("count", 0)),
        "customers_available": bool(customers.get("available")),
        "customer_count": int(customers.get("customer_count", len(customer_table) if isinstance(customer_table, pd.DataFrame) else 0)),
        "discounts": discounts,
        "daily_rows": int(len(daily)),
        "daily_sales_file": "daily_sales.csv",
        "sales_kpis": {
            "aov": round(float(sales.get("aov", 0)), 2),
            "growth": sales.get("growth"),
            "date_start": sales.get("date_start"),
            "date_end": sales.get("date_end"),
        },
    }
    _write_json(output_dir / "analytics_summary.json", summary)
    return summary


def load_daily_sales(output_dir: str | Path) -> pd.DataFrame:
    """Load Stage 2's canonical daily revenue dataset for forecasting."""
    path = Path(output_dir) / "daily_sales.csv"
    if not path.is_file():
        raise FileNotFoundError("daily_sales.csv is missing. Generate Stage 2 analytics first.")

    daily = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "revenue", "orders"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError("daily_sales.csv is missing columns: " + ", ".join(sorted(missing)))

    daily["revenue"] = pd.to_numeric(daily["revenue"], errors="coerce")
    daily["orders"] = pd.to_numeric(daily["orders"], errors="coerce").fillna(0)
    daily = daily.dropna(subset=["date", "revenue"]).sort_values("date")
    daily = daily.drop_duplicates(subset=["date"], keep="last").set_index("date")
    if daily.empty:
        raise ValueError("daily_sales.csv contains no valid daily observations.")

    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0)
    daily.index.name = "date"
    daily["revenue"] = daily["revenue"].astype(float)
    daily["orders"] = daily["orders"].astype(int)
    return daily.reset_index()


def read_report(output_dir: str | Path, filename: str) -> dict:
    """Read a generated JSON report, returning an empty object if absent."""
    path = Path(output_dir) / filename
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
