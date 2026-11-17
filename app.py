"""Flask web app: serves the HTML/CSS/JS frontend and a small JSON API around the trained models."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
app = Flask(__name__)


ADVICE = {
    "good": "Air quality is satisfactory and poses little or no risk. A good day for outdoor activity.",
    "satisfactory": "Acceptable for most people. Unusually sensitive people may notice minor breathing discomfort.",
    "moderate": "People with lung or heart disease, children and older adults should reduce long outdoor exertion.",
    "very poor": "Avoid outdoor exercise. Keep windows closed, use an air purifier indoors and wear an N95 mask outside.",
    "poor": "Limit prolonged outdoor activity. Sensitive groups should stay indoors where possible.",
    "severe": "Health emergency. Stay indoors, avoid all outdoor exertion and follow local health advisories.",
    "low": "Pollution is on the lower side for this dataset.",
    "medium": "Pollution is at a medium level for this dataset. Sensitive groups should take care.",
    "high": "Pollution is high for this dataset. Limit outdoor exposure.",
}


def advice_for(label):
    key = str(label).strip().lower()
    return ADVICE.get(key, "")


def load_artifacts():
    paths = [os.path.join(MODEL_DIR, n) for n in ("regressor.joblib", "classifier.joblib", "metrics.json")]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    with open(paths[2]) as fh:
        metrics = json.load(fh)
    return joblib.load(paths[0]), joblib.load(paths[1]), metrics


REG, CLS, METRICS = load_artifacts()


@app.route("/")
def index():
    return render_template("index.html", ready=METRICS is not None)


@app.route("/api/metrics")
def api_metrics():
    if METRICS is None:
        return jsonify(error="Models not trained yet. Run: python train.py --data data/AirQualityData.csv"), 503
    return jsonify(METRICS)



@app.route("/api/predict", methods=["POST"])
def api_predict():
    if METRICS is None:
        return jsonify(error="Models not trained yet. Run train.py first."), 503
    payload = (request.get_json(silent=True) or {}).get("features", {})
    schema = METRICS["schema"]
    numeric = {c["name"] for c in schema["numeric"]}
    row = {}
    for f in METRICS["dataset"]["features"]:
        v = payload.get(f)
        if f in numeric or f in ("Month", "DayOfWeek"):
            if v in (None, ""):
                row[f] = np.nan
            else:
                try:
                    row[f] = float(v)
                except (TypeError, ValueError):
                    return jsonify(error=f"'{f}' must be a number."), 400
                if row[f] < 0:
                    return jsonify(error=f"'{f}' cannot be negative."), 400
        else:
            row[f] = v if v else np.nan
    frame = pd.DataFrame([row])[METRICS["dataset"]["features"]]
    value = float(np.clip(REG.predict(frame)[0], 0, 500 if "aqi" in METRICS["dataset"]["target"].lower() else 1e6))
    proba = CLS.predict_proba(frame)[0]
    order = np.argsort(proba)[::-1]
    label = str(CLS.classes_[order[0]])
    return jsonify(
        value=round(value, 1),
        target=METRICS["dataset"]["target"],
        category=label,
        confidence=round(float(proba[order[0]]) * 100, 1),
        probabilities=[{"label": str(CLS.classes_[i]), "p": round(float(proba[i]) * 100, 1)} for i in order[:4]],
        advice=advice_for(label),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
