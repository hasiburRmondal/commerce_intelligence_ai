
#!/usr/bin/env python3
"""WooCommerce CSV diagnostic v2: mixed date formats + product-header discovery.

Run:
    python data_cleaning_test_v2.py "orders.csv"

Requires:
    pip install pandas

Reads in chunks; does not modify the CSV or print customer/order data.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

NUMERIC = [
    "shipping_total",
    "fee_total",
    "tax_total",
    "cart_discount",
    "order_discount",
    "discount_total",
    "discount_tax",
    "order_total",
    "order_subtotal"
]


def parse_dates(series):
    raw = series.astype(str).str.strip()

    out = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    nonblank = raw.ne("")

    # ISO-like dates: YYYY-MM-DD, optionally with time.
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}([ T].*)?$")

    if iso.any():
        out.loc[iso] = pd.to_datetime(
            raw.loc[iso],
            errors="coerce",
            format="mixed"
        )

    # Day-first dates: DD-MM-YYYY or DD/MM/YYYY.
    dmy = ~iso & raw.str.match(
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}([ T].*)?$"
    )

    if dmy.any():
        out.loc[dmy] = pd.to_datetime(
            raw.loc[dmy],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

    # Fallback for other valid formats.
    other = nonblank & out.isna() & ~iso & ~dmy

    if other.any():
        out.loc[other] = pd.to_datetime(
            raw.loc[other],
            errors="coerce",
            format="mixed"
        )

    return out, nonblank


def num(series):
    """Convert numeric values while preserving negative amounts."""

    raw = series.astype(str).str.strip()

    # Remove leading apostrophe from values such as '-28.99
    raw = raw.str.replace(r"^'", "", regex=True)

    # Remove thousands separators.
    raw = raw.str.replace(",", "", regex=False)

    # Convert to numeric.
    return pd.to_numeric(
        raw.replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA
        }),
        errors="coerce"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("csv_file", nargs="?")
    ap.add_argument("--chunksize", type=int, default=5000)

    args = ap.parse_args()

    filename = (
        args.csv_file
        or input("CSV path: ").strip().strip('"')
    )

    path = Path(filename).expanduser()

    if not path.is_file():
        sys.exit(f"File not found: {path}")

    # Detect CSV delimiter.
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:

        sample = f.read(100000)

    try:
        delimiter = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|"
        ).delimiter

    except csv.Error:
        delimiter = ","

    print(
        "\n" + "=" * 68
        + "\nWOOCOMMERCE DATA DIAGNOSTIC V2\n"
        + "=" * 68
    )

    print(
        f"File: {path.name} | "
        f"Size: {path.stat().st_size / 1048576:.2f} MB"
    )

    print(
        f"Delimiter: {repr(delimiter)} | "
        f"Chunk size: {args.chunksize:,}"
    )

    try:
        reader = pd.read_csv(
            path,
            sep=delimiter,
            encoding="utf-8-sig",
            dtype=str,
            chunksize=args.chunksize,
            keep_default_na=False,
            on_bad_lines="warn",
            low_memory=False
        )

    except Exception as e:
        sys.exit(
            f"CSV open error: {type(e).__name__}: {e}"
        )

    headers = None
    rows = 0

    statuses = Counter()
    missing = Counter()

    date_bad = Counter()
    date_nonblank = Counter()

    numeric = {
        c: [0, 0, 0.0]
        for c in NUMERIC
    }

    ids_seen = set()

    duplicate_ids = 0
    blank_ids = 0

    product_rows = 0
    missing_sku = 0
    missing_name = 0

    product_qty = 0.0
    product_total = 0.0
    product_subtotal = 0.0

    product_total_n = 0
    product_subtotal_n = 0

    pack_offers = Counter()
    bundle_count = 0

    product_header_candidates = []

    for chunk_i, df in enumerate(reader, 1):

        df.columns = [
            str(c).replace("\ufeff", "").strip()
            for c in df.columns
        ]

        rows += len(df)

        if headers is None:

            headers = list(df.columns)

            product_header_candidates = [
                c for c in headers
                if re.search(
                    r"product|line.?item|sku|variation|bundle|offer",
                    c,
                    re.I
                )
            ]

            print(f"Columns detected: {len(headers)}")

            print(
                "\nPRODUCT-RELATED HEADER NAMES "
                "(names only; safe to share):"
            )

            if product_header_candidates:

                for c in product_header_candidates:
                    print("  " + repr(c))

            else:
                print(
                    "  NONE MATCHED product/line-item/"
                    "SKU/variation/bundle/offer"
                )

        # Required fields.
        for c in [
            "order_id",
            "order_date",
            "status",
            "order_total"
        ]:

            if c in df:

                missing[c] += int(
                    df[c].astype(str).str.strip().eq("").sum()
                )

            else:

                missing["MISSING:" + c] += len(df)

        # Order statuses.
        if "status" in df:

            statuses.update(
                df["status"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("", "(blank)")
            )

        # Duplicate and blank order IDs.
        if "order_id" in df:

            for oid in df["order_id"].astype(str).str.strip():

                if not oid:
                    blank_ids += 1

                elif oid in ids_seen:
                    duplicate_ids += 1

                else:
                    ids_seen.add(oid)

        # Date validation.
        for c in [
            "order_date",
            "paid_date",
            "date_completed"
        ]:

            if c in df:

                parsed, nonblank = parse_dates(df[c])

                date_nonblank[c] += int(nonblank.sum())

                date_bad[c] += int(
                    (nonblank & parsed.isna()).sum()
                )

        # Numeric validation.
        for c in NUMERIC:

            if c in df:

                raw = df[c].astype(str).str.strip()

                values = num(df[c])

                numeric[c][0] += int(
                    values.notna().sum()
                )

                numeric[c][1] += int(
                    (raw.ne("") & values.isna()).sum()
                )

                numeric[c][2] += float(
                    values.sum(skipna=True)
                )

        # Match structured product headers.
        normalized = {
            re.sub(r"\s+", " ", c).strip().lower(): c
            for c in df.columns
        }

        name_keys = []

        for key in normalized:

            m = re.fullmatch(
                r"product item\s*(\d+)\s*name",
                key
            )

            if m:
                name_keys.append(
                    (int(m.group(1)), normalized[key])
                )

        for slot, name_col in sorted(name_keys):

            def get(suffix):

                key = f"product item {slot} {suffix}".lower()

                col = normalized.get(key)

                if col:
                    return df[col]

                return pd.Series(
                    [""] * len(df),
                    index=df.index
                )

            names = get("name").astype(str).str.strip()

            skus = get("sku").astype(str).str.strip()

            q = num(get("quantity"))
            t = num(get("total"))
            st = num(get("subtotal"))

            active = (
                names.ne("")
                | skus.ne("")
                | q.notna()
                | t.notna()
                | st.notna()
            )

            product_rows += int(active.sum())

            missing_sku += int(
                (active & skus.eq("")).sum()
            )

            missing_name += int(
                (active & names.eq("")).sum()
            )

            product_qty += float(
                q[active].sum(skipna=True)
            )

            product_total += float(
                t[active].sum(skipna=True)
            )

            product_subtotal += float(
                st[active].sum(skipna=True)
            )

            product_total_n += int(
                t[active].notna().sum()
            )

            product_subtotal_n += int(
                st[active].notna().sum()
            )

        # Scan line-item metadata.
        line_cols = [
            c for c in df.columns
            if re.search(r"line.?item", c, re.I)
        ]

        for c in line_cols:

            for value in df[c].astype(str):

                if not value:
                    continue

                m = re.search(
                    r"_tvw_selected_offer:([^|]+)",
                    value,
                    re.I
                )

                if m:
                    pack_offers[m.group(1).strip()] += 1

                if "_tvw_bundle_item:" in value:
                    bundle_count += 1

        if chunk_i % 10 == 0:
            print(f"Scanned {rows:,} rows...")

    # Final report.
    print(
        "\n" + "-" * 68
        + "\nSUMMARY\n"
        + "-" * 68
    )

    print(
        f"Rows: {rows:,} | "
        f"Columns: {len(headers or []):,}"
    )

    print(f"Blank required fields: {dict(missing)}")

    print(
        f"Blank order IDs: {blank_ids:,} | "
        f"Duplicate order IDs: {duplicate_ids:,}"
    )

    print("\nSTATUS COUNTS")

    for k, v in statuses.most_common():
        print(f"{k:<18} {v:>10,}")

    print("\nDATE PARSING")

    for c in [
        "order_date",
        "paid_date",
        "date_completed"
    ]:

        if c in date_nonblank:

            print(
                f"{c}: "
                f"nonblank={date_nonblank[c]:,}, "
                f"invalid={date_bad[c]:,}"
            )

    print("\nNUMERIC FIELDS")

    for c, (valid, bad, total) in numeric.items():

        if c in (headers or []):

            print(
                f"{c}: valid={valid:,}, "
                f"invalid_nonblank={bad:,}, "
                f"sum={total:,.2f}"
            )

    print("\nSTRUCTURED PRODUCT COLUMNS")

    print(f"Product lines: {product_rows:,}")

    print(
        f"Missing SKU: {missing_sku:,} | "
        f"Missing name: {missing_name:,}"
    )

    print(f"Quantity sum: {product_qty:,.2f}")

    print(
        f"Line total sum: {product_total:,.2f} "
        f"(cells={product_total_n:,})"
    )

    print(
        f"Line subtotal sum: {product_subtotal:,.2f} "
        f"(cells={product_subtotal_n:,})"
    )

    print("\nPACK / BUNDLE METADATA")

    print(
        f"Bundle metadata occurrences: {bundle_count:,}"
    )

    for k, v in pack_offers.most_common(20):
        print(f"{k}: {v:,}")

    print(
        "\nIMPORTANT: Share this console output only, "
        "not the CSV."
    )

    print("No source data was changed.")


if __name__ == "__main__":
    main()
