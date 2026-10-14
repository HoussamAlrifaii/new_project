# BikePulse: understanding training, models, and results

## Yes, we trained the models

The cloud execution ran `python -m bikepulse.train`, and the EDA notebook also ran the same training function. Training is computation performed by the Python code; it does not require manually pressing a button in a notebook. The files contain an actual trained experiment, not prewritten or assumed results.

`bikepulse/model.py` contains the feature engineering and training logic. `bikepulse/train.py` invokes it and saves a fitted model, predictions, and metrics locally under `artifacts/`. Those generated artifacts are ignored by Git. The smaller results and provenance summaries in `reports/` are intended for version control. The app fits and caches its model on its first run; it does not load the saved pickle file.

## What training means

A model receives examples of inputs and known answers. Here an input contains calendar attributes and past rental counts, and the answer is the next hour's total rentals. Fitting adjusts model parameters so its predictions approximate the observed answers.

We cannot measure generalization by testing on the same examples used to fit or select a model. A model might memorize or exploit patterns that do not transfer. For forecasting, we also must preserve time order: training on later dates and testing on earlier ones would not represent real forecasting.

## Our chronological experiment

| Stage | Dates | Eligible rows | What happens |
|---|---|---:|---|
| Training | Before July 1, 2012 | 12,591 | Fit Ridge and gradient boosting |
| Validation | July–September 2012 | 2,208 | Compare models and choose lower MAE |
| Refit | Training plus validation dates | 14,799 | Fit the chosen model again using all pre-test data |
| Test | October–December 2012 | 2,098 | Evaluate selected model on held-out future dates |

The test set was not used to choose between Ridge and gradient boosting. Ridge's validation result and the selected model's test result refer to different periods, so they should not be compared as though they were a head-to-head test comparison.

The experiment evaluates repeated **one-hour-ahead** forecasts. At each forecast hour, actual demand from prior hours is available as input, including prior hours inside the test period. This is legitimate under the stated forecast contract: those observations have already happened. It is not a prediction of the entire three-month period made in September, and it does not demonstrate a 24-hour rollout.

All predictions are clipped at zero because negative rental counts are not meaningful. The same evaluated predictions are used in the results and replay interface.

## MAE: mean absolute error

For each forecast, subtract predicted rentals from observed rentals, take the absolute value, then average over the evaluated hours:

`MAE = mean(abs(actual - predicted))`

Example: actual values [100, 200, 300] and predictions [110, 180, 330] give absolute errors [10, 20, 30]. MAE is 20 rentals/hour.

Lower is better on the same target, rows and period. A test MAE of 28.3 means the mean absolute error across our 2,098 eligible test hours is 28.3 rentals. It is not 28.3%, not classification accuracy, and not a guarantee that every prediction is within 28.3 rentals.

## RMSE: root mean squared error

Square each error, average the squared errors, then take the square root:

`RMSE = sqrt(mean((actual - predicted)**2))`

In the example above, RMSE is approximately 21.6 rentals/hour. Squaring gives larger errors more influence. An error of 100 contributes 10,000 before the square root, whereas an error of 10 contributes 100.

MAE summarizes typical absolute error in original units. RMSE is more sensitive to occasional large misses. Here RMSE is 45.1 while MAE is 28.3, showing that larger errors matter. To understand where those misses occur, examine the notebook's error-by-hour chart and individual prediction rows. Neither metric by itself measures bias, uncertainty coverage, or operational cost.

## Ridge regression

Ordinary linear regression models a prediction as an intercept plus a weighted sum of input features. Ridge adds a penalty for large coefficient values:

`objective = sum of squared prediction errors + alpha * sum of squared coefficients`

The penalty, called L2 regularization, encourages smaller weights and can make predictions more stable when inputs are related or noisy. `alpha` controls the penalty strength. A larger value increases regularization; it does not automatically produce a better model. Our first version uses alpha=10 without a broad search.

We encode hour, weekday, month and holiday as categories using one-hot encoding, and standardize the three demand lags. One-hot encoding avoids assuming that each one-hour increase has the same linear effect. Standardization puts lag variables on a comparable scale for the coefficient penalty.

Ridge is a useful interpretable benchmark. However, the current additive formulation has limited ability to represent interactions such as a different rush-hour shape on weekends unless we explicitly engineer interaction features.

## Gradient boosting

Gradient boosting combines many small decision trees. It starts with a simple prediction and sequentially adds trees that reduce remaining loss. For squared-error regression, this can be understood as successive corrections to residual mistakes.

A tree can split on rules such as hour, weekday, and recent demand. Combining trees captures nonlinear behavior and feature interactions without requiring every interaction to be manually specified.

Our model is scikit-learn's `HistGradientBoostingRegressor`: a histogram-based implementation. We use 150 boosting iterations, at most 15 leaves per tree, L2 regularization of 10, and a fixed random seed. Internal random validation is disabled so we use our explicit chronological validation set. These are initial settings, not proof of globally optimal tuning.

Boosting is flexible, but it can still overfit, struggle with changing conditions, and make large errors during unusual events. A complicated model is useful only if evaluation justifies it.

## What the results actually say

### Validation: selecting a model

| Model | MAE, rentals/hour | RMSE, rentals/hour |
|---|---:|---:|
| Ridge | 52.18 | 75.75 |
| Gradient boosting | 35.40 | 54.45 |

Gradient boosting has lower validation MAE, so the pipeline selects it. This supports choosing it for this experiment; it does not prove Ridge is always worse.

### Test: selected model versus the weekly baseline

The weekly baseline simply predicts the observed rental count at the same hour seven days earlier. It needs no training and provides a useful seasonal reference.

| Predictor | MAE, rentals/hour | RMSE, rentals/hour |
|---|---:|---:|
| Selected gradient boosting model, refitted | 28.31 | 45.13 |
| Same-hour-last-week baseline | 69.56 | 114.85 |

MAE reduction is `(69.5639 - 28.3127) / 69.5639 * 100`, approximately **59.3%**. This is an improvement over this baseline, not 59.3% accuracy. The test period has different seasonal conditions from validation; a lower test error does not prove performance is improving over time.

The initial replay date also shows why we need to inspect individual errors: at 08:00 on October 15, 2012, the model predicted approximately 470 rentals and observed demand was 737. A good overall average can coexist with a substantial rush-hour miss.

## Are other models necessarily better?

No. Other candidates may improve performance, but we cannot claim that before comparing them under the same conditions. Data quality, feature design, forecast horizon, and validation often matter more than a fashionable algorithm.

Useful next candidates include:

1. Last-hour persistence and hour-by-weekday averages as additional simple baselines.
2. Ridge with explicit hour-by-weekday interactions.
3. Random forests, Extra Trees, or tuned histogram gradient boosting.
4. XGBoost, LightGBM or CatBoost, if extra dependencies are justified.
5. Count-oriented objectives and richer lag or rolling-window features, constructed strictly from past observations.

A neural network or LSTM is not automatically preferable for a small tabular dataset. We should first run rolling-origin validation across several time windows, use that validation to tune settings, and report accuracy, peak-hour error, runtime and complexity. Repeatedly choosing new models by their October–December test score would turn that set into another validation set; a new untouched final test period would then be needed.

## What has been verified and what remains

Verified: real data downloaded, training completed, metrics independently recalculated, notebook executed, interactive app tested, and two automated tests passed. Exact timestamp lags and current-target leakage are tested.

Remaining: recent-data validation, additional baselines, systematic tuning, uncertainty calibration, and public deployment. This release is a historical portfolio demonstration, not a validated present-day operational service.
