# Air Quality Prediction using Machine Learning

Predicts the **AQI value** (regression) and the **AQI category** (classification) from pollutant readings,
with a web frontend built in HTML, CSS and JavaScript on top of a Flask API.

## Run it
```bash
pip install -r requirements.txt
```

1. Open `air_quality_ml.ipynb` in Jupyter or VS Code and run its cells from top to bottom. It trains and evaluates the models, then saves model files and charts for the app. Change `DATA_PATH` in the notebook to use another CSV.
2. Run `python app.py` and open http://127.0.0.1:5000.

The command-line training script is also available: `python train.py --data data/AirQualityData.csv`. If the target column is not detected automatically, add `--target <column name>`.

The included `data/AirQualityData.csv` contains 4,000 hourly records. The notebook and CLI both default to this file; change `DATA_PATH` in the notebook or pass `--data <path>` to use another CSV.

## Structure
| File | Purpose |
|---|---|
| `air_quality_ml.ipynb` | interactive data preparation, EDA, model training, evaluation, and artifact generation |
| `train.py` | command-line version of the training pipeline |
| `app.py` | Flask server + `/api/predict` and `/api/metrics` |
| `templates/index.html`, `static/style.css`, `static/script.js` | frontend |
| `models/` | `regressor.joblib`, `classifier.joblib`, `metrics.json` (created by the notebook or train.py) |
| `static/charts/` | charts used in the web app, report and PPT |

## Dashboard features
- Compare prediction scenarios using typical or higher-reading presets.
- View training dataset summaries, model scores, and charts produced by the notebook or `train.py`.
- Keep the latest 20 predictions in browser-local storage; export them as CSV or clear them from the History tab.
- Download the model metrics and dataset schema as a JSON report.

Prediction history is stored only in the current browser and is not sent to the Flask server. Predictions are estimates from the trained dataset, not live monitoring or official health guidance.

## Method
1. Clean: parse dates, exclude target leakage and invalid negative sensor readings, then impute missing numeric and categorical values.
2. Features: use available pollutant and environmental readings, categorical columns, and date-derived month and weekday features.
3. Models: Linear/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
4. Evaluate on an 80/20 split (R², RMSE, MAE; accuracy, weighted F1) plus 5-fold cross-validation for the best regressor.
5. Classify using a provided AQI category when available, CPCB bands for AQI targets, or target terciles otherwise. Permutation importance summarizes influential input features.
