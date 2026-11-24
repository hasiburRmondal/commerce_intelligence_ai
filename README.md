# Commerce Intelligence AI

A local, CSV-powered WooCommerce analytics and demand forecasting application built with Flask, Pandas, scikit-learn, Jinja2 and Chart.js. It does not connect to WooCommerce, use a database, or call an external AI API.

## Features

- Upload WooCommerce order CSV exports (up to 120 MB).
- Parse order dates, statuses, totals and Product Item 1–60 line-item columns.
- Sales KPIs, daily/monthly revenue charts and period comparison.
- Product performance table and chart.
- Customer RFM-style segments using customer ID/user ID only.
- Order discount summary.
- Seasonal Naive, Random Forest and HistGradientBoosting comparison with chronological holdout; MAE, RMSE and WAPE.
- Recursive 7/30/60/90-day store-level sales forecast.
- Local rule-based analytics assistant.
- Daily sales and product CSV exports.

## Requirements

Python 3.10–3.12 recommended.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Expected CSV columns

Required: `order_id`, `order_date`, `status`. Sales totals use `order_total`; discounts use `discount_total`. Product parsing supports `Product Item N Name`, `Product Item N id`, `Product Item N SKU`, `Product Item N Quantity`, `Product Item N Total`, and `Product Item N Subtotal`, for N=1..60. Customer analysis uses `customer_id` or `customer_user`. Missing optional columns are handled.

## Important data assumptions

- By default, `completed` and `processing` orders are included. If neither status exists in the file, a fallback excludes cancelled, failed, refunded, trash and draft statuses. Review this policy for your store before interpreting totals.
- Order-level revenue uses `order_total`. Product revenue uses line-item `Product Item N Total`. These can differ because of shipping, tax, discounts and refunds.
- Duplicate order IDs are reduced to one order row (last occurrence) to avoid double-counting order totals. Product line items are extracted from the export rows and filtered to accepted order IDs.
- This first version expects the structured `Product Item N ...` columns. It does not parse serialized `line_item_N` blobs.
- Customer IDs are used for segmentation and are still pseudonymous personal data. Keep the CSV on a trusted machine and remove uploads when finished.
- The app stores the uploaded CSV temporarily in `uploads/`. Use the **Remove uploaded data** button when done. Do not expose this development server publicly.
- Chart.js is loaded from a CDN, so charts require internet access in the browser. All business data processing and ML run locally.
- The local assistant is rule-based, not a generative AI model.
- A single year of data is limited for annual seasonality; forecasts are estimates. Product-level forecasts and stock/reorder advice are not claimed in this version.

## Configuration

Set `SECRET_KEY` as an environment variable before deployment. Set `debug=False` and use a production WSGI server for any non-development environment. This project is designed for local use, not public multi-user hosting.


## CSV cleaning updates

The CSV reader now removes leading apostrophes from numeric values while
preserving negative amounts. For example, `'-28.99` becomes `-28.99`.
Negative `fee_total` values remain negative custom-discount adjustments.

It also parses mixed date formats, includes `fee_total` and other monetary
fields in the normalized order data, and reports invalid non-empty numeric
values in `data['numeric_diagnostics']`.

The product export quantity is treated as individual units, as confirmed for
this project. Pack metadata must not multiply this quantity.

The application remains CSV-only: no WooCommerce REST API, database, or
external AI API is required.

Run the regression tests from the project root:

```bash
pytest -q
```
