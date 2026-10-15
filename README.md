# BikePulse — Urban Bike Demand Forecasting 🚲

A real-data, one-hour-ahead Capital Bikeshare demand forecasting project with Python EDA, chronological model selection, and a Streamlit historical replay interface.

[Open the Colab notebook](https://colab.research.google.com/github/HoussamAlrifaii/new_project/blob/main/notebooks/BikePulse_Colab.ipynb) — save a copy in Drive, then run the setup cell first.

## Learn how the project works

- [Data dictionary and chart explanations](docs/01_data_and_charts.md)
- [Training, MAE/RMSE, models and results](docs/02_training_and_results.md)
- [GitHub title and publication status](docs/03_github_and_project_status.md)
- [Step-by-step VS Code, training and EDA guide](docs/04_run_in_vscode.md)

The dataset covers 2011–2012. This release demonstrates historical forecasting; present-day usefulness requires recent data and new validation.

## Run locally (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m bikepulse.train
python -m pytest -q
streamlit run app.py
```

On Windows activate `.venv\Scripts\activate` and omit the shell environment variable prefixes. Open `notebooks/01_eda_and_modeling.ipynb` in VS Code and select the `.venv` interpreter. Run every cell in order.

The first run downloads the authentic UCI hourly CSV over verified HTTPS. Internet access to `archive.ics.uci.edu` is required. Local data, virtual environments, and trained models are ignored by Git. Data provenance and SHA-256 fingerprints are saved in `data/provenance.json`; fingerprints record the downloaded bytes, not independent authenticity verification.

## Forecast contract

Predict total system rentals for an observed hour using hour, weekday, month, holiday, and demand exactly 1, 24, and 168 hours earlier. Demand is assumed available at the end of the preceding hour. These lags use timestamps rather than row positions because hours are missing in the source. Rows without all required observations are excluded.

Train on data before July 2012; choose Ridge or gradient boosting using July–September MAE. Refit the selected model through September and evaluate once on October–December. Compare with demand at the same hour a week earlier. Test forecasts update lags with actual preceding observations, so this evaluates repeated one-hour forecasts rather than a multi-hour rollout. `casual`, `registered`, and observed future weather are excluded to avoid leakage.

`artifacts/metrics.json` records MAE, RMSE, row counts and the selected model. The validated historical test results are recorded in `reports/metrics.json`: gradient boosting MAE 28.3 rentals/hour and RMSE 45.1 versus weekly-baseline MAE 69.6 and RMSE 114.9, across 2,098 eligible test hours. This is a 59.3% MAE reduction. The EDA notebook has been executed top-to-bottom. The app shows a fixed held-out historical replay; it does not predict live station inventory. No causal claims or calibrated prediction intervals are made. Holiday information is limited to the historical dataset.

## Source and attribution

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T and Gama, DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Real Capital Bikeshare observations from Washington, D.C., 2011–2012. This repository does not redistribute the raw data.

## Deploy

Push the project to GitHub, then create a Streamlit Community Cloud app selecting `app.py` and Python 3.12. The host must reach the UCI download URL; the app downloads data and fits its cached model on first startup. This is intended for a small portfolio demo. Do not load model pickle files from untrusted sources. Public hosting and GitHub push have not been performed by the build script.

## Project layout

- `bikepulse/data.py`: verified-HTTPS ingestion and structural checks
- `bikepulse/model.py`: exact-hour features, training and evaluation
- `bikepulse/train.py`: reproducible artifact generation
- `notebooks/01_eda_and_modeling.ipynb`: EDA and experiment companion
- `app.py`: interactive historical replay and demand exploration
- `tests/test_model.py`: missing-hour alignment and future-target leakage checks

## Next research steps

Add rolling-origin evaluation across seasons, inspect peak-demand errors, compare more recent data, and assess uncertainty coverage. Station availability requires real inventory snapshots; trip records alone are insufficient.
