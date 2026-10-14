# BikePulse: project title and GitHub status

## Recommended identity

Project title: **BikePulse — Urban Bike Demand Forecasting**

Recommended GitHub repository name: **bikepulse-demand-forecasting**

Recommended description: **Real-data hourly bike-demand forecasting with Python EDA, chronological ML evaluation, and a Streamlit historical replay app.**

## GitHub publication and naming

Project files were committed and pushed to GitHub `main` on October 14, 2026. They are also available in the cloud checkout at `/workspace/new_project`. The Git origin currently uses `HoussamAlrifaii/new_project`. The project title is BikePulse; the remote repository name is a separate GitHub setting. Use `git log` and the GitHub repository page to inspect published commits. Public application hosting is separate from publishing source code.

GitHub API access returned `Forbidden`, so the API route cannot rename the repository. The recommended `bikepulse-demand-forecasting` URL was also not accessible at the latest check. The local folder remains `new_project` to preserve saved setup paths.

## Rename the GitHub repository

On GitHub, open `HoussamAlrifaii/new_project`, go to **Settings → General → Repository name**, enter `bikepulse-demand-forecasting`, and choose **Rename**. This is the external action needed because the current API route is unavailable.

After the rename is confirmed, the checkout origin can be updated to:

```bash
git remote set-url origin https://github.com/HoussamAlrifaii/bikepulse-demand-forecasting.git
```

After renaming, verify the remote with `git ls-remote origin`. Raw data, virtual environments, logs and model artifacts are excluded by `.gitignore`. Do not force-push or overwrite another branch when publishing changes.

## Read the project in order

1. `docs/01_data_and_charts.md`: dataset, every original column, missing hours and chart interpretation.
2. `docs/02_training_and_results.md`: actual training, chronological evaluation, MAE, RMSE, Ridge, gradient boosting and limitations.
3. `notebooks/01_eda_and_modeling.ipynb`: executed evidence with tables and charts.
4. `README.md`: run and deployment instructions.

`reports/eda_notebook.html` is a readable export of the same executed notebook; it is not a separate experiment. `reports/eda_notebook.md` is another export. Their results intentionally match.

Public Streamlit hosting has not yet been configured. The current app runs in the development environment. A future public demo should retain the historical-data notice rather than imply that it predicts live station availability.
