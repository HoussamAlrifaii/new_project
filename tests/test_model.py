import pandas as pd
from bikepulse.model import features, prepare, FEATURES


def test_lags_use_exact_hours_and_past_only():
    index = pd.date_range('2012-01-01', periods=200, freq='h').delete(20)
    frame = pd.DataFrame({'cnt': range(len(index)), 'holiday': 0}, index=index)
    x = features(frame)
    assert pd.isna(x.loc['2012-01-01 21:00', 'lag_1'])
    t = index[-1]
    assert x.loc[t, 'lag_24'] == frame.loc[t - pd.Timedelta(hours=24), 'cnt']
    changed = frame.copy()
    changed.loc[t:, 'cnt'] = 99999
    pd.testing.assert_series_equal(features(changed).loc[t], x.loc[t])
    assert list(prepare(frame)[0].columns) == FEATURES
    assert not {'cnt', 'casual', 'registered', 'temp', 'atemp'} & set(FEATURES)
