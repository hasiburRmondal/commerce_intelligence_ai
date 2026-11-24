"""WooCommerce CSV ingestion and cleaning. CSV-only; no API or database."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

REQUIRED = ["order_id", "order_date", "status"]
MONEY_COLUMNS = [
    "order_total", "discount_total", "order_subtotal", "shipping_total",
    "fee_total", "tax_total", "cart_discount", "order_discount", "discount_tax",
]
PRODUCT_FIELDS = {
    "name": "product_name", "id": "product_id", "sku": "sku",
    "quantity": "quantity", "total": "line_total", "subtotal": "line_subtotal",
}


def _detect_delimiter(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(100_000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _number(series):
    """Parse numbers; remove CSV apostrophe prefixes but preserve minus signs."""
    raw = series.astype("string").str.strip()
    raw = raw.str.replace(r"^'+", "", regex=True)
    raw = raw.str.replace(",", "", regex=False)
    raw = raw.str.replace(r"^[₹$€£]\s*", "", regex=True)
    raw = raw.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "-": pd.NA})
    return pd.to_numeric(raw, errors="coerce")


def _parse_dates(series):
    raw = series.astype("string").str.strip()
    raw = raw.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    try:
        return pd.to_datetime(raw, format="mixed", errors="coerce", dayfirst=True)
    except (TypeError, ValueError):
        iso = raw.str.match(r"^\d{4}-\d{1,2}-\d{1,2}", na=False)
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if iso.any():
            result.loc[iso] = pd.to_datetime(raw.loc[iso], errors="coerce", yearfirst=True)
        other = raw.notna() & ~iso
        if other.any():
            result.loc[other] = pd.to_datetime(raw.loc[other], errors="coerce", dayfirst=True)
        return result


def _product_column_map(columns):
    found = {}
    aliases = {
        "name": {"name", "product name", "item name"},
        "id": {"id", "product id", "product"},
        "sku": {"sku", "product sku"},
        "quantity": {"quantity", "qty"},
        "total": {"total", "line total"},
        "subtotal": {"subtotal", "sub total", "line subtotal"},
    }
    for column in columns:
        match = re.match(r"(?i)^product\s*item\s*(\d+)\s*(.*?)\s*$", str(column).strip())
        if not match:
            continue
        slot = int(match.group(1))
        suffix = re.sub(r"[^a-z0-9]+", " ", match.group(2).lower()).strip()
        for key, names in aliases.items():
            if suffix in names:
                found.setdefault(slot, {})[key] = column
                break
    return found


def load_orders_csv(path):
    """Return normalized orders/items plus safe data-quality diagnostics."""
    path = Path(path)
    delimiter = _detect_delimiter(path)
    df = pd.read_csv(
        path, sep=delimiter, encoding="utf-8-sig", dtype="string",
        keep_default_na=False, low_memory=False,
    )
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    raw_rows = len(df)
    blank_required = {
        c: int(df[c].astype("string").str.strip().eq("").sum()) for c in REQUIRED
    }
    duplicate_orders = int(df.duplicated("order_id").sum())
    blank_order_ids = int(df["order_id"].astype("string").str.strip().eq("").sum())

    for col in ["order_date", "paid_date", "date_completed"]:
        if col in df:
            df[col] = _parse_dates(df[col])
    missing_dates = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date"]).copy()
    df["status"] = df["status"].astype("string").str.lower().str.strip()

    numeric_diagnostics = {}
    for col in MONEY_COLUMNS:
        if col not in df:
            df[col] = 0.0
            numeric_diagnostics[col] = {"valid": 0, "invalid_nonblank": 0, "sum": 0.0}
            continue
        original = df[col].astype("string").str.strip()
        parsed = _number(df[col])
        invalid = original.ne("") & parsed.isna()
        numeric_diagnostics[col] = {
            "valid": int(parsed.notna().sum()),
            "invalid_nonblank": int(invalid.sum()),
            "sum": float(parsed.sum(skipna=True)),
        }
        # Blank monetary cells become zero for dashboard arithmetic.
        # Invalid non-empty cells remain NaN and are counted above.
        df[col] = parsed.where(original.ne(""), 0.0)

    # Preserve the original app's order-status policy.
    valid_status = {"completed", "processing"}
    orders = df[df["status"].isin(valid_status)].copy()
    if orders.empty:
        orders = df[~df["status"].isin(
            {"cancelled", "failed", "refunded", "trash", "draft"}
        )].copy()
    orders = orders.drop_duplicates(subset=["order_id"], keep="last")

    product_map = _product_column_map(df.columns)
    item_frames = []
    for slot, fields in sorted(product_map.items()):
        part = pd.DataFrame(index=df.index)
        for field, output in PRODUCT_FIELDS.items():
            col = fields.get(field)
            part[output] = df[col] if col else ("" if field in {"name", "id", "sku"} else pd.NA)
        part["order_id"] = df["order_id"]
        part["order_date"] = df["order_date"]
        part["item_number"] = slot

        for col in ["quantity", "line_total", "line_subtotal"]:
            part[col] = _number(part[col])
        for col in ["product_name", "product_id", "sku"]:
            part[col] = part[col].fillna("").astype("string").str.strip()

        active = part["product_name"].ne("") | part["sku"].ne("") | part["product_id"].ne("")
        part = part[active & part["quantity"].gt(0)].copy()
        if not part.empty:
            # Confirmed project rule: exported quantity already means individual units.
            # Pack labels are not multiplied into quantity.
            part["quantity_individual_units"] = part["quantity"]
            item_frames.append(part[[
                "order_id", "order_date", "item_number", "product_id", "sku",
                "product_name", "quantity", "quantity_individual_units",
                "line_total", "line_subtotal",
            ]])

    item_columns = [
        "order_id", "order_date", "item_number", "product_id", "sku",
        "product_name", "quantity", "quantity_individual_units",
        "line_total", "line_subtotal",
    ]
    items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame(columns=item_columns)
    if not items.empty:
        items = items[items["order_id"].isin(orders["order_id"])].copy()

    # Do not pass customer names, emails, phones, addresses, IPs, user agents,
    # transaction IDs, or order notes into analysis services.
    safe_order_columns = [
        "order_id", "order_number", "order_date", "paid_date", "date_completed",
        "status", "currency",
        # Keep non-contact customer identifiers for RFM analysis.
        "customer_id", "customer_user",
        *MONEY_COLUMNS,
    ]
    orders = orders[[c for c in safe_order_columns if c in orders.columns]].copy()

    missing_sku_items = (
        int(items["sku"].fillna("").astype(str).str.strip().eq("").sum())
        if not items.empty else 0
    )
    return {
        "orders": orders,
        "items": items,
        "columns": list(df.columns),
        "raw_rows": raw_rows,
        "duplicate_orders": duplicate_orders,
        "blank_order_ids": blank_order_ids,
        "blank_required": blank_required,
        "missing_dates": missing_dates,
        "numeric_diagnostics": numeric_diagnostics,
        "product_slots_detected": len(product_map),
        "missing_sku_items": missing_sku_items,
    }
