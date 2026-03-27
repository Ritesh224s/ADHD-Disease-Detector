"""
NeuroScan AI - ADHD Detection Flask Backend
"""

import os
import sys
import pickle
import json
import gc
import webbrowser
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, make_response

# Suppress Flask warnings
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Fix UTF-8 encoding on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    import numpy as np
    import pandas as pd
    from flask import Flask, request, jsonify, send_from_directory
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run: pip install flask numpy pandas scikit-learn")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model_xgb = None
scaler = None

# Load XGBoost Model
try:
    with open(os.path.join(BASE_DIR, "xgboost_adhd_model.pkl"), "rb") as f:
        model_xgb = pickle.load(f)
    # Use CPU for predictions
    try:
        model_xgb.set_params(predictor="cpu_predictor")
    except:
        pass
    print("[OK] XGBoost model loaded (CPU mode for Flask compatibility)")
except FileNotFoundError:
    print("[ERROR] XGBoost model file not found!")
except Exception as e:
    print(f"[ERROR] Failed to load XGBoost model: {e}")

# Load Scaler
try:
    with open(os.path.join(BASE_DIR, "xgboost_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print("[OK] Scaler loaded")
except Exception as e:
    print(f"[WARNING] Scaler not found: {e}")

EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/sample-data", methods=["GET"])
def get_sample_data():
    """Get a random sample from the sample patients dataset for testing"""
    try:
        # Get category from query parameters if needed
        # It defaults to 'random' if not specified
        category = request.args.get("category", "random")
        sample_file_path = os.path.join(BASE_DIR, "sample_patients.json")

        sample = None
        if os.path.exists(sample_file_path):
            with open(sample_file_path, "r") as f:
                data = json.load(f)

            categories = list(data.keys())
            if category in categories:
                # Pick a random sample from the requested category
                import random

                sample = random.choice(data[category])
                selected_category = category
            else:
                # Pick a completely random category and a random sample from it
                import random

                selected_category = random.choice(categories)
                sample = random.choice(data[selected_category])
                print(f"[INFO] Using random category: {selected_category}")

            # Fill missing channels with 0.0 to match EEG_CHANNELS
            for channel in EEG_CHANNELS:
                if channel not in sample:
                    sample[channel] = 0.0

        if sample is None:
            # Absolute fallback if file doesn't exist
            print(
                "[WARNING] sample_patients.json not found. Using fallback sample data."
            )
            sample = {
                "Fp1": 19.74,
                "Fp2": 4.25,
                "F3": 14.84,
                "F4": -53.32,
                "F7": 69.12,
                "F8": 7.60,
                "Fz": -17.36,
                "C3": 0.76,
                "C4": -18.93,
                "Cz": -27.41,
                "T7": 37.43,
                "T8": -39.00,
                "P3": 37.15,
                "P4": -36.92,
                "P7": -88.06,
                "P8": 45.17,
                "Pz": 20.89,
                "O1": 55.32,
                "O2": 96.56,
            }
            selected_category = "fallback"
            for channel in EEG_CHANNELS:
                if channel not in sample:
                    sample[channel] = 0.0

        return jsonify(
            {
                "success": True,
                "sample": sample,
                "category": selected_category,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        print(f"Error getting sample data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/<path:filename>")
def serve_static(filename):
    if filename.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Process EEG data for ADHD analysis"""
    try:
        if model_xgb is None or scaler is None:
            return jsonify(
                {"error": "Model not loaded. Server configuration issue."}
            ), 500

        mode = request.form.get("mode", "upload").lower().strip()
        data_dict = {}
        data_points = "1"

        # ========== Manual Input Mode ==========
        if mode == "manual":
            # Collect all 19 EEG channel values from form
            for channel in EEG_CHANNELS:
                try:
                    value = request.form.get(channel.lower(), "0")
                    data_dict[channel] = float(value) if value else 0.0
                except (ValueError, TypeError):
                    print(f"[WARNING] Invalid value for {channel}: {value}")
                    data_dict[channel] = 0.0

            # Ensure all channels are present
            for channel in EEG_CHANNELS:
                if channel not in data_dict:
                    data_dict[channel] = 0.0

        # ========== File Upload Mode ==========
        else:
            if "file" not in request.files:
                return jsonify(
                    {"error": "No file provided. Please upload a CSV file."}
                ), 400

            file = request.files["file"]
            if not file or not file.filename:
                return jsonify(
                    {"error": "No file selected. Please choose a CSV file."}
                ), 400

            # Validate file extension
            if not (
                file.filename.lower().endswith(".csv")
                or file.filename.lower().endswith(".txt")
            ):
                return jsonify(
                    {"error": "Invalid file type. Please upload a CSV or TXT file."}
                ), 400

            try:
                # Read CSV file
                df = pd.read_csv(file, encoding="utf-8", nrows=1)  # Read only first row

                if df.empty:
                    return jsonify(
                        {"error": "CSV file is empty. Please provide valid data."}
                    ), 400

                # Extract first row
                row = df.iloc[0]

                # Collect channel data from CSV
                missing_channels = []
                for channel in EEG_CHANNELS:
                    if channel in df.columns:
                        try:
                            value = float(row[channel])
                            data_dict[channel] = value
                        except (ValueError, TypeError):
                            print(
                                f"[WARNING] Non-numeric value for {channel}: {row[channel]}"
                            )
                            data_dict[channel] = 0.0
                    else:
                        missing_channels.append(channel)
                        data_dict[channel] = 0.0

                # Log missing channels (but don't fail)
                if missing_channels:
                    print(f"[INFO] Missing channels in CSV: {missing_channels}")

                # Reset file pointer before reading again
                file.seek(0)

                # Count total data points in file
                total_rows = len(pd.read_csv(file, encoding="utf-8"))
                data_points = str(total_rows)

            except pd.errors.ParserError as pe:
                return jsonify(
                    {"error": f"CSV parsing error: Invalid file format."}
                ), 400
            except Exception as e:
                print(f"[ERROR] File processing error: {str(e)}")
                return jsonify({"error": f"Failed to process file: {str(e)}"}), 400

        # ========== Make Prediction ==========
        try:
            # Create feature DataFrame with proper column order
            feature_values = [data_dict.get(ch, 0.0) for ch in EEG_CHANNELS]
            X = pd.DataFrame([feature_values], columns=EEG_CHANNELS)

            # Scale features
            X_scaled = scaler.transform(X)

            # Get prediction and probability with error handling
            try:
                predictions = model_xgb.predict(X_scaled)
                pred = int(predictions[0])

                # Get confidence score
                try:
                    proba = model_xgb.predict_proba(X_scaled)
                    conf = float(max(proba[0]) * 100)
                except:
                    # If predict_proba fails, estimate confidence from prediction
                    conf = 75.0  # Default confidence if unavailable

            except Exception as pred_error:
                print(f"[ERROR] Prediction failed: {str(pred_error)}")
                # Try to recover with a simpler prediction
                try:
                    pred = int(model_xgb.predict(X_scaled)[0])
                    conf = 50.0  # Default confidence
                except:
                    return jsonify(
                        {"error": "Prediction engine failed. Please try again."}
                    ), 500

            # Prepare response
            response_data = {
                "success": True,
                "classification": "ADHD Detected" if pred == 1 else "Control (Normal)",
                "classType": "adhd" if pred == 1 else "control",
                "confidence": round(conf, 2),
                "dataPoints": data_points,
                "timestamp": datetime.now().isoformat(),
            }

            return jsonify(response_data), 200

        except Exception as pred_error:
            print(f"[ERROR] Prediction error: {str(pred_error)}")
            import traceback

            traceback.print_exc()
            return jsonify({"error": "Prediction failed. Please try again."}), 500

    except Exception as e:
        print(f"[ERROR] Analyze endpoint error: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    finally:
        # Explicit memory cleanup
        gc.collect()


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    """Generate a clinical report PDF"""
    try:
        from io import BytesIO

        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
                PageBreak,
            )
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
        except ImportError:
            print("[WARNING] reportlab not installed. Using alternative method.")
            return jsonify(
                {
                    "success": False,
                    "error": "PDF generation library not available. Please install reportlab.",
                    "install_command": "pip install reportlab",
                }
            ), 500

        # Get data from request
        data = request.get_json() or {}
        patient_id = data.get("patientId", "N/A")
        patient_age = data.get("patientAge", "N/A")
        classification = data.get("classification", "N/A")
        confidence = data.get("confidence", 0)
        timestamp = data.get("timestamp", datetime.now().isoformat())
        eeg_data = data.get("eegData", {})

        # Create PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title Style
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#0891b2"),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )

        subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#155e75"),
            spaceAfter=8,
            fontName="Helvetica-Bold",
        )

        # Header
        story.append(Paragraph("ADHD Clinical Assessment Report", title_style))
        story.append(
            Paragraph("MediNeuro AI - Clinical Decision Support System", subtitle_style)
        )
        story.append(Spacer(1, 0.2 * inch))

        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ["Patient ID:", patient_id],
            ["Age (years):", patient_age],
            ["Assessment Date:", timestamp.split("T")[0]],
            [
                "Assessment Time:",
                timestamp.split("T")[1][:5] if "T" in timestamp else "N/A",
            ],
        ]

        patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfeff")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ]
            )
        )
        story.append(patient_table)
        story.append(Spacer(1, 0.3 * inch))

        # Assessment Results
        story.append(Paragraph("Assessment Results", heading_style))
        result_color = (
            colors.HexColor("#16a34a")
            if classification == "Control (Normal)"
            else colors.HexColor("#dc2626")
        )

        result_data = [
            ["Classification:", classification],
            ["Model Confidence:", f"{confidence}%"],
            ["Model Type:", "XGBoost Ensemble"],
            ["Algorithm Version:", "2.0 (CPU)"],
        ]

        result_table = Table(result_data, colWidths=[2 * inch, 4 * inch])
        result_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfeff")),
                    (
                        "BACKGROUND",
                        (1, 0),
                        (1, 0),
                        colors.HexColor("#fef2f2")
                        if "ADHD" in classification
                        else colors.HexColor("#f0fdf4"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (1, 0), (1, 0), result_color),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ]
            )
        )
        story.append(result_table)
        story.append(Spacer(1, 0.3 * inch))

        # EEG Channels Data
        if eeg_data:
            story.append(Paragraph("EEG Signal Data (Microvolts)", heading_style))

            # Create two-column layout for EEG data
            eeg_rows = [["Channel", "Value (μV)", "Channel", "Value (μV)"]]
            channels_list = list(eeg_data.items())

            for i in range(0, len(channels_list), 2):
                ch1, val1 = channels_list[i]
                ch2_val = (
                    channels_list[i + 1] if i + 1 < len(channels_list) else ("", "")
                )
                if ch2_val:
                    ch2, val2 = ch2_val
                    eeg_rows.append(
                        [ch1.upper(), f"{val1:.2f}", ch2.upper(), f"{val2:.2f}"]
                    )
                else:
                    eeg_rows.append([ch1.upper(), f"{val1:.2f}", "", ""])

            eeg_table = Table(
                eeg_rows, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
            )
            eeg_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0891b2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#ecfeff")],
                        ),
                    ]
                )
            )
            story.append(eeg_table)
            story.append(Spacer(1, 0.3 * inch))

        # Clinical Notes
        story.append(Paragraph("Clinical Notes", heading_style))
        notes = [
            "• This assessment is based on quantitative EEG (qEEG) signal analysis using advanced machine learning.",
            "• The XGBoost ensemble model was trained on clinical EEG datasets with 70.63% accuracy.",
            "• This report should be used as a screening aid and NOT as a standalone diagnostic tool.",
            "• Professional clinical evaluation and diagnosis must be performed by a qualified healthcare provider.",
            "• Results should be interpreted in conjunction with clinical history, symptoms, and other diagnostic measures.",
        ]

        for note in notes:
            story.append(Paragraph(note, styles["Normal"]))

        story.append(Spacer(1, 0.3 * inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#dc2626"),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            borderColor=colors.HexColor("#fecaca"),
            borderWidth=1,
            borderPadding=8,
        )

        disclaimer_text = (
            "DISCLAIMER: This AI-based assessment is for clinical support purposes only and should not be used as the sole basis for diagnosis. "
            "A qualified healthcare professional must review these results within the context of the patient's complete clinical presentation. "
            "The system developers and operators assume no liability for clinical decisions made based on this analysis."
        )

        story.append(Paragraph(disclaimer_text, disclaimer_style))

        # Footer
        story.append(Spacer(1, 0.4 * inch))
        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#94a3b8"),
            alignment=TA_CENTER,
        )
        story.append(
            Paragraph(
                "© 2026 MediNeuro AI • Clinical Decision Support System", footer_style
            )
        )
        story.append(
            Paragraph(
                "Report Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                footer_style,
            )
        )

        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)

        # Create response
        response = make_response(pdf_buffer.getvalue())
        response.headers["Content-Disposition"] = (
            f'attachment; filename="ADHD_Assessment_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
        )
        response.headers["Content-Type"] = "application/pdf"
        return response

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify(
            {"success": False, "error": f"PDF generation failed: {str(e)}"}
        ), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "XGBoost CPU"})


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    print(f"Server error: {error}")
    return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🏥 MediNeuro AI - Clinical ADHD Diagnosis Support System")
    print("=" * 70)

    if model_xgb is None or scaler is None:
        print("[CRITICAL] Models not loaded. Cannot start server.")
        print("Please ensure these files exist:")
        print(f"  - {os.path.join(BASE_DIR, 'xgboost_adhd_model.pkl')}")
        print(f"  - {os.path.join(BASE_DIR, 'xgboost_scaler.pkl')}")
        sys.exit(1)

    print("\n✅ All systems operational")
    print("🌐 Launching web interface...")
    print("⏳ Opening in browser...\n")

    # Open browser after a short delay to ensure Flask has started
    def open_browser():
        import time

        time.sleep(2)
        try:
            webbrowser.open("http://127.0.0.1:5000")
            print("✅ Browser opened successfully\n")
        except Exception as e:
            print(f"ℹ️  Manual access: http://127.0.0.1:5000\n")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        app.run(
            debug=False, host="127.0.0.1", port=5000, use_reloader=False, threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n[✓] Server stopped")
    except Exception as main_err:
        print(f"\n[ERROR] Fatal error: {main_err}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
