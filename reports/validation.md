# Validation

Assessment: share with caveats as a historical portfolio demonstration.

- Source: authentic UCI Capital Bikeshare hourly CSV, 2011–2012; 17,379 observations. Download fingerprints are retained in `data_provenance.json`.
- Structural checks: unique timestamps, nonnegative targets, casual + registered = total. Missing clock hours are not filled with invented observations.
- Features: calendar attributes and exact past-clock-hour lags. The automated test verifies missing-hour alignment and that changing the current target cannot change current predictors.
- Selection: train before July 2012; compare Ridge and gradient boosting on July–September; refit the selected model through September. Held-out October–December test has 2,098 eligible observations.
- Calculation: MAE recomputed independently from saved prediction rows, matching training output; 28.3127 versus weekly baseline 69.5639 rentals/hour, a 59.3% reduction.
- Notebook: all eight code cells executed top-to-bottom; charts and tables are embedded.
- Interface: Streamlit AppTest rendered without exceptions and successfully changed the forecast hour. Local server health endpoint returned `ok`.

Limitations: rolling one-hour forecasts use actual preceding observations, including earlier test-period demand; this is expected at each forecast origin and is not a multi-hour rollout. Hours without lag observations are excluded. Dataset timestamps use source clock labels without timezone localization; daylight-saving transitions are not modeled separately. Results are historical, system-wide, and do not establish station availability or present-day performance. No calibrated intervals or causal claims. Source code has been pushed to GitHub main. Public application hosting remains outstanding.
