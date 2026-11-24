def test_services_import():
    from services.csv_reader import load_orders_csv
    from services.sales_analysis import build_sales_analysis
    from services.forecasting import run_forecasting
    assert callable(load_orders_csv) and callable(build_sales_analysis) and callable(run_forecasting)
