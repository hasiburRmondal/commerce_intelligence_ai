"""Privacy-conscious customer RFM analysis using customer IDs only."""
import pandas as pd


def build_customer_analysis(orders):
    """Build RFM segments, preferring customer_id and falling back to customer_user.

    Customer names, emails, phone numbers, IP addresses and billing/shipping
    addresses are not required or displayed by this analysis.
    """
    required = {"order_id", "order_date", "order_total"}
    if not required.issubset(orders.columns):
        return {
            "available": False,
            "segments": {},
            "table": pd.DataFrame(),
            "message": "The uploaded data is missing fields required for RFM analysis.",
        }

    data = orders.copy()

    # Use customer_id when populated; otherwise fall back row-by-row to customer_user.
    customer_id = (
        data["customer_id"].astype("string").str.strip()
        if "customer_id" in data.columns
        else pd.Series(pd.NA, index=data.index, dtype="string")
    )
    customer_user = (
        data["customer_user"].astype("string").str.strip()
        if "customer_user" in data.columns
        else pd.Series(pd.NA, index=data.index, dtype="string")
    )

    customer_id = customer_id.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "0": pd.NA})
    customer_user = customer_user.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "0": pd.NA})

    data["customer_reference"] = customer_id.fillna(customer_user)
    data = data.dropna(subset=["customer_reference"]).copy()

    if data.empty:
        return {
            "available": False,
            "segments": {},
            "table": pd.DataFrame(),
            "message": (
                "Customer ID values are empty. Include customer_id or "
                "customer_user in your WooCommerce CSV export."
            ),
        }

    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["order_total"] = pd.to_numeric(data["order_total"], errors="coerce")
    data = data.dropna(subset=["order_date", "order_total"])

    if data.empty:
        return {
            "available": False,
            "segments": {},
            "table": pd.DataFrame(),
            "message": "No valid customer order dates or order totals were found.",
        }

    now = data["order_date"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        data.groupby("customer_reference", dropna=True)
        .agg(
            recency=("order_date", lambda values: (now - values.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("order_total", "sum"),
        )
        .reset_index()
    )

    # Quantile scores. Rank first to avoid qcut duplicate-edge failures.
    for metric, ascending in [
        ("recency", False),
        ("frequency", True),
        ("monetary", True),
    ]:
        if len(rfm) <= 1:
            rfm[f"{metric}_score"] = 1
        else:
            ranks = rfm[metric].rank(method="first", ascending=ascending)
            bins = min(4, len(rfm))
            rfm[f"{metric}_score"] = (
                pd.qcut(ranks, q=bins, labels=False, duplicates="drop").astype(int) + 1
            )

    rfm["segment"] = "Regular"
    if len(rfm):
        median_monetary = rfm["monetary"].median()
        rfm.loc[
            (rfm["frequency"] >= 2) & (rfm["monetary"] >= median_monetary),
            "segment",
        ] = "Loyal"
        rfm.loc[
            (rfm["recency"] > 60) & (rfm["frequency"] >= 2),
            "segment",
        ] = "At Risk"
        rfm.loc[rfm["frequency"] == 1, "segment"] = "One-time"

    counts = rfm["segment"].value_counts().to_dict()
    return {
        "available": True,
        "segments": counts,
        "table": rfm.sort_values("monetary", ascending=False).head(100),
        "customer_count": int(len(rfm)),
        "message": None,
    }
