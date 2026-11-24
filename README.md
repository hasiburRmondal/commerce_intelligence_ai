# Commerce Intelligence AI

A local, CSV-powered **WooCommerce Analytics and Machine Learning Forecasting Application** built with Python, Flask, Pandas, Scikit-learn, Jinja2, and Chart.js.

Commerce Intelligence AI transforms exported WooCommerce order data into actionable business insights through sales analytics, product performance analysis, customer segmentation, and machine learning-based demand forecasting.

**Designed for local execution:** No WooCommerce REST API, external database, or external AI API is required.

---

## Key Features

### 1. WooCommerce CSV Processing
- Upload WooCommerce order CSV exports (up to 120 MB).
- Parse order dates, statuses, order totals, discounts, and fees.
- Extract product information from structured `Product Item 1–60` columns.
- Normalize numeric values and handle mixed date formats.
- Generate cleaning diagnostics for invalid numeric values.

### 2. Sales Analytics
- Total revenue and key business performance indicators.
- Daily and monthly sales analysis.
- Revenue charts and period comparisons.
- Order discount summaries.
- Daily sales CSV export.

### 3. Product Performance Analysis
- Product-level sales and revenue analysis.
- Product quantity and performance summaries.
- Product performance tables and visualizations.
- Product CSV export.

### 4. Customer Segmentation
- RFM-style customer segmentation.
- Customer analysis using customer ID or user ID.
- Recency, frequency, and monetary-based insights.
- Customer identification through IDs rather than exposing customer contact information.

### 5. Machine Learning-Based Sales Forecasting

Compare multiple forecasting approaches using historical WooCommerce sales data.

**Implemented models:**
- Seasonal Naive
- Random Forest Regressor
- HistGradientBoosting Regressor

**Model evaluation metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)

**Forecasting capabilities:**
- Chronological train/test evaluation.
- Model performance comparison.
- Recursive sales forecasting.
- 7-day, 30-day, 60-day, and 90-day forecast horizons.

### 6. Local Analytics Assistant
- Rule-based analytics assistant.
- Answers predefined business questions using locally processed analytics.
- Does not require an external generative AI service.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Frontend | HTML, CSS, JavaScript, Jinja2 |
| Data Visualization | Chart.js |
| Data Input | WooCommerce CSV Export |
| Testing | Pytest |

---

## Application Workflow

```text
WooCommerce CSV Export
        |
        v
CSV Upload and Validation
        |
        v
Data Cleaning and Normalization
        |
        v
Data Processing with Pandas
        |
        +--------------------------+
        |                          |
        v                          v
   Sales Analytics          Product Analytics
        |                          |
        v                          v
   Customer Segmentation    Product Performance
        |
        v
Sales Forecasting
        |
        v
Model Comparison and Evaluation
        |
        v
7 / 30 / 60 / 90-Day Forecast
        |
        v
Flask Dashboard and Reports
```

---

## Installation and Setup

### Requirements

- Python 3.10–3.12 recommended.
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/commerce-intelligence-ai.git
cd commerce-intelligence-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

## Expected CSV Format

The application expects a WooCommerce order export containing the following fields.

### Required columns

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `order_date` | Order creation date |
| `status` | WooCommerce order status |

### Sales and discount columns

| Column | Description |
|---|---|
| `order_total` | Total order value |
| `discount_total` | Total order discount |

### Product line-item columns

The application supports structured product columns from `Product Item 1` through `Product Item 60`.

Examples:

```text
Product Item 1 Name
Product Item 1 id
Product Item 1 SKU
Product Item 1 Quantity
Product Item 1 Total
Product Item 1 Subtotal
```

### Customer identification

Customer analysis uses either:

```text
customer_id
customer_user
```

Missing optional columns are handled by the application.

---

## Data Cleaning and Processing

The CSV processing pipeline includes several normalization steps:

- Supports mixed date formats.
- Converts monetary fields into numeric values.
- Removes leading apostrophes from numeric values while preserving negative amounts.
- Handles missing and invalid numeric values through diagnostics.
- Processes order and product line-item data separately.
- Includes fee and other monetary fields in normalized order data.

**Negative fee values are preserved.** For example:

```text
'-28.99  ->  -28.99
```

Negative `fee_total` values represent custom discount adjustments in this dataset and are not converted into positive amounts.

### Product quantity handling

Product quantities are interpreted as individual units.

Pack metadata, such as `3pc`, `By Box(5)`, or `Value Pack(15)`, is not used to multiply the exported product quantity.

---

## Important Data Assumptions and Limitations

### Order status handling
- By default, `completed` and `processing` orders are included.
- If neither status exists, a fallback excludes cancelled, failed, refunded, trash, and draft statuses.
- Review the order status policy before interpreting revenue figures.

### Revenue calculations
- Order-level revenue uses `order_total`.
- Product-level revenue uses `Product Item N Total`.
- These values may differ because of shipping, taxes, discounts, and refunds.

### Duplicate orders
- Duplicate order IDs are reduced to one order row using the last occurrence to avoid double-counting order totals.
- Product line items are extracted from export rows and filtered against accepted order IDs.

### Product parsing
- The current implementation expects structured `Product Item N ...` columns.
- Serialized `line_item_N` data is not parsed.

### Forecasting limitations
- Forecasts are based on historical store-level sales data.
- A single year of historical data limits the ability to learn annual seasonal patterns.
- Forecasts are estimates, not guaranteed future sales.
- Product-level forecasting and automatic stock/reorder recommendations are not implemented in this version.

---

## Privacy and Security

This application is designed for local processing of WooCommerce exports.

- Customer names, email addresses, phone numbers, billing addresses, shipping addresses, IP addresses, and other unnecessary personal information are not required for the analytics outputs.
- Customer IDs used for segmentation are still pseudonymous personal data.
- Uploaded CSV files are temporarily stored in the `uploads/` directory.
- Use the **Remove uploaded data** button when finished.
- Keep exported business data on a trusted machine.
- Do not expose the Flask development server publicly.

**External services:** Business data processing and ML execution run locally. Chart.js is loaded from a CDN, so internet access is required in the browser to render charts.

---

## Configuration

Before using the application outside local development:

- Configure `SECRET_KEY` through an environment variable.
- Set `debug=False`.
- Use a production WSGI server.
- Configure appropriate upload restrictions and server security.

This project is designed for local use, not public multi-user hosting.

---

## Running Tests

Run the regression test suite from the project root:

```bash
pytest -q
```

---

## Project Objectives

This project demonstrates practical application of:

- Python-based data processing.
- Data cleaning and exploratory data analysis.
- Business analytics and KPI reporting.
- Customer segmentation.
- Supervised machine learning and regression.
- Time-series forecasting techniques.
- Model comparison and evaluation.
- Flask-based application development.
- Interactive data visualization.

## Future Improvements

Potential future development areas:

- More robust rolling-origin forecasting evaluation.
- SKU-level demand forecasting.
- Additional forecasting algorithms.
- Forecast uncertainty estimation.
- Automated data quality reporting.
- Model monitoring and performance tracking.
- Improved inventory planning capabilities.

---
