from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_real_data_replay_and_interaction():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics['Observed rentals'] == '737'
    assert float(metrics['Test MAE (rentals per hour)']) > 0
    before = metrics['Predicted rentals']
    app.slider[0].set_value(17).run()
    assert not app.exception
    after = {item.label: item.value for item in app.metric}
    assert after['Predicted rentals'] != before
