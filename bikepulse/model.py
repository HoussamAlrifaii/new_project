import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CALENDAR = ['hour', 'weekday', 'month', 'holiday']
FEATURES = CALENDAR + ['lag_1', 'lag_24', 'lag_168']


def features(frame):
    """Exact clock-hour lags; missing hours are never treated as adjacent."""
    result = pd.DataFrame(index=frame.index)
    result['hour'] = frame.index.hour
    result['weekday'] = frame.index.dayofweek
    result['month'] = frame.index.month
    result['holiday'] = frame['holiday']
    for lag in (1, 24, 168):
        result[f'lag_{lag}'] = frame['cnt'].reindex(
            frame.index - pd.Timedelta(hours=lag)).to_numpy()
    return result


def prepare(frame):
    x = features(frame).dropna()
    return x, frame.loc[x.index, 'cnt']


def candidates():
    transform = ColumnTransformer([
        ('calendar', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CALENDAR),
        ('lags', StandardScaler(), ['lag_1', 'lag_24', 'lag_168'])
    ])
    return {
        'Ridge': make_pipeline(transform, Ridge(alpha=10)),
        'Gradient boosting': HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, l2_regularization=10,
            early_stopping=False, random_state=42)
    }


def score(y, prediction):
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction)))}


def train(frame):
    x, y = prepare(frame)
    training = x.index < '2012-07-01'
    validation = (x.index >= '2012-07-01') & (x.index < '2012-10-01')
    testing = x.index >= '2012-10-01'
    models = candidates()
    validation_scores = {}
    for name, model in models.items():
        model.fit(x.loc[training], y.loc[training])
        validation_scores[name] = score(y.loc[validation], np.maximum(0, model.predict(x.loc[validation])))
    selected = min(validation_scores, key=lambda name: validation_scores[name]['MAE'])
    model = models[selected]
    model.fit(x.loc[~testing], y.loc[~testing])
    prediction = np.maximum(0, model.predict(x.loc[testing]))
    baseline = x.loc[testing, 'lag_168'].to_numpy()
    metrics = {
        'selected_model': selected,
        'validation': validation_scores,
        'test_model': score(y.loc[testing], prediction),
        'test_weekly_baseline': score(y.loc[testing], baseline),
        'rows': {'train': int(training.sum()), 'validation': int(validation.sum()), 'test': int(testing.sum())},
        'test_start': '2012-10-01',
        'evaluation': 'Rolling one-hour-ahead evaluation using actual demand from preceding hours.'
    }
    predictions = pd.DataFrame({'actual': y.loc[testing], 'prediction': prediction, 'weekly_baseline': baseline})
    return model, metrics, predictions
