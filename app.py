from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import uuid

from config import Config
from services.pipeline import (
    clean_and_validate,
    generate_analytics,
    load_cleaned_data,
    read_report,
)
from services.sales_analysis import build_sales_analysis, daily_sales_frame
from services.product_analysis import build_product_analysis
from services.customer_analysis import build_customer_analysis
from services.discount_analysis import build_discount_analysis
from services.forecasting import run_forecasting
from services.assistant import answer_question
from services.exports import dataframe_csv_response


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["ARTIFACT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    def upload_path():
        value = session.get("csv_path")
        path = Path(value) if value else None
        return path if path and path.is_file() else None

    def artifact_dir():
        upload_id = session.get("upload_id")
        if not upload_id:
            return None
        path = Path(app.config["ARTIFACT_FOLDER"]) / upload_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def current_data(require_analytics=True):
        folder = artifact_dir()
        if folder is None:
            return None
        if require_analytics and not session.get("analytics_ready"):
            return None
        try:
            return load_cleaned_data(folder)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return None

    # Pandas is imported here for the exception type used by current_data.
    import pandas as pd

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Choose a WooCommerce CSV file.", "error")
                return redirect(url_for("index"))
            if Path(file.filename).suffix.lower() != ".csv":
                flash("Only CSV files are supported.", "error")
                return redirect(url_for("index"))

            upload_id = uuid.uuid4().hex
            filename = f"{upload_id}.csv"
            path = Path(app.config["UPLOAD_FOLDER"]) / filename
            file.save(path)

            # New upload invalidates the previous upload's artifacts and session state.
            old_path = upload_path()
            old_upload_id = session.get("upload_id")
            if old_path and old_path != path:
                old_path.unlink(missing_ok=True)
            if old_upload_id:
                import shutil
                shutil.rmtree(Path(app.config["ARTIFACT_FOLDER"]) / old_upload_id, ignore_errors=True)

            session.clear()
            session["csv_path"] = str(path)
            session["source_name"] = secure_filename(file.filename)
            session["upload_id"] = upload_id
            session["cleaning_ready"] = False
            session["analytics_ready"] = False
            session["forecast_ready"] = False

            return redirect(url_for("processing"))

        return render_template("upload.html")

    @app.route("/processing")
    def processing():
        if not upload_path():
            return redirect(url_for("index"))
        folder = artifact_dir()
        report = read_report(folder, "cleaning_report.json")
        summary = read_report(folder, "analytics_summary.json")
        return render_template(
            "processing.html",
            source=session.get("source_name"),
            cleaning_ready=session.get("cleaning_ready", False),
            analytics_ready=session.get("analytics_ready", False),
            forecast_ready=session.get("forecast_ready", False),
            cleaning_report=report,
            analytics_summary=summary,
        )

    @app.route("/process/clean", methods=["POST"])
    def process_clean():
        source = upload_path()
        if source is None:
            flash("Upload a CSV before starting data cleaning.", "error")
            return redirect(url_for("index"))
        try:
            report = clean_and_validate(source, artifact_dir())
            session["cleaning_ready"] = True
            session["analytics_ready"] = False
            session["forecast_ready"] = False
            flash(
                f"Cleaning completed: {report['clean_order_rows']:,} orders and "
                f"{report['product_line_rows']:,} product lines normalized.",
                "success",
            )
        except Exception as exc:
            flash(f"CSV cleaning failed: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("processing"))

    @app.route("/process/analytics", methods=["POST"])
    def process_analytics():
        if not session.get("cleaning_ready"):
            flash("Complete Stage 1 before generating analytics.", "error")
            return redirect(url_for("processing"))
        try:
            summary = generate_analytics(artifact_dir())
            session["analytics_ready"] = True
            session["forecast_ready"] = False
            flash(
                f"Analytics generated: {summary['orders']:,} orders, "
                f"{summary['products']:,} products and {summary['daily_rows']:,} daily rows.",
                "success",
            )
        except Exception as exc:
            flash(f"Analytics generation failed: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("processing"))

    @app.route("/process/forecast", methods=["POST"])
    def process_forecast():
        if not session.get("analytics_ready"):
            flash("Complete Stage 2 before running forecasting.", "error")
            return redirect(url_for("processing"))
        try:
            data = current_data()
            horizon = 30
            result = run_forecasting(data["orders"], horizon, app.config["ARTIFACT_FOLDER"])
            if not result.get("available"):
                raise ValueError(result.get("message", "Forecasting could not run."))
            folder = artifact_dir()
            (folder / "forecast_30.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            session["forecast_ready"] = True
            flash("Initial 30-day forecast generated successfully.", "success")
        except Exception as exc:
            flash(f"Forecasting failed: {type(exc).__name__}: {exc}", "error")
        return redirect(url_for("processing"))

    def require_analytics():
        if not upload_path():
            return redirect(url_for("index"))
        if not session.get("cleaning_ready"):
            flash("Run Stage 1: Clean & Validate CSV first.", "error")
            return redirect(url_for("processing"))
        if not session.get("analytics_ready"):
            flash("Run Stage 2: Generate Analytics first.", "error")
            return redirect(url_for("processing"))
        return None

    @app.route("/dashboard")
    def dashboard():
        gate = require_analytics()
        if gate:
            return gate
        data = current_data()
        result = build_sales_analysis(data["orders"], data["items"])
        return render_template("dashboard.html", result=result, source=session.get("source_name"))

    @app.route("/products")
    def products():
        gate = require_analytics()
        if gate:
            return gate
        result = build_product_analysis(current_data()["items"])
        return render_template("products.html", result=result)

    @app.route("/customers")
    def customers():
        gate = require_analytics()
        if gate:
            return gate
        result = build_customer_analysis(current_data()["orders"])
        return render_template("customers.html", result=result)

    @app.route("/discounts")
    def discounts():
        gate = require_analytics()
        if gate:
            return gate
        result = build_discount_analysis(current_data()["orders"])
        return render_template("discounts.html", result=result)

    @app.route("/forecast", methods=["GET", "POST"])
    def forecast():
        gate = require_analytics()
        if gate:
            return gate
        data = current_data()
        horizon = int(request.form.get("horizon", 30)) if request.method == "POST" else 30
        horizon = horizon if horizon in (7, 30, 60, 90) else 30
        result = run_forecasting(data["orders"], horizon, app.config["ARTIFACT_FOLDER"])
        if result.get("available"):
            session["forecast_ready"] = True
        return render_template("forecast.html", result=result, horizon=horizon)

    @app.route("/assistant", methods=["GET", "POST"])
    def assistant():
        gate = require_analytics()
        if gate:
            return gate
        data = current_data()
        answer = None
        question = ""
        if request.method == "POST":
            question = request.form.get("question", "").strip()
            answer = answer_question(question, data)
        return render_template("assistant.html", answer=answer, question=question)

    @app.route("/export/forecast-backtest")
    def export_forecast_backtest():
        gate = require_analytics()
        if gate:
            return gate

        data = current_data()
        try:
            horizon = int(request.args.get("horizon", 30))
        except (TypeError, ValueError):
            horizon = 30
        horizon = horizon if horizon in (7, 30, 60, 90) else 30

        result = run_forecasting(
            data["orders"],
            horizon,
            app.config["ARTIFACT_FOLDER"],
        )
        if not result.get("available") or not result.get("backtest_rows"):
            flash("No forecast backtesting rows are available to export.", "error")
            return redirect(url_for("forecast"))

        frame = pd.DataFrame(result["backtest_rows"])
        csv_data = frame.to_csv(index=False)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=forecast_backtesting_{horizon}_days.csv"
                )
            },
        )

    @app.route("/export/<kind>")
    def export(kind):
        gate = require_analytics()
        if gate:
            return gate
        data = current_data()
        if kind == "daily":
            return dataframe_csv_response(daily_sales_frame(data["orders"]), "daily_sales.csv")
        if kind == "products":
            return dataframe_csv_response(
                build_product_analysis(data["items"])["table"],
                "product_performance.csv",
            )
        if kind == "cleaning-report":
            from flask import send_file
            path = artifact_dir() / "cleaning_report.json"
            return send_file(path, as_attachment=True, download_name="cleaning_report.json")
        flash("Unknown export type.", "error")
        return redirect(url_for("dashboard"))

    @app.route("/remove-data", methods=["POST"])
    def remove_data():
        path = session.pop("csv_path", None)
        upload_id = session.pop("upload_id", None)
        session.clear()
        if path:
            Path(path).unlink(missing_ok=True)
        if upload_id:
            import shutil
            shutil.rmtree(Path(app.config["ARTIFACT_FOLDER"]) / upload_id, ignore_errors=True)
        flash("Uploaded CSV and generated artifacts removed from this application.", "success")
        return redirect(url_for("index"))

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", message="File exceeds the 120 MB upload limit."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
