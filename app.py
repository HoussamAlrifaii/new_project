import pandas as pd
import streamlit as st
from bikepulse.data import load_data
from bikepulse.model import FEATURES, prepare, train

st.set_page_config(page_title='BikePulse', page_icon='🚲', layout='wide')
st.title('🚲 BikePulse')
st.write('Explore real Capital Bikeshare demand and replay one-hour-ahead forecasts.')
st.caption('Washington, D.C. · 2011–2012 · Historical demonstration, not a live station-availability service.')

@st.cache_data
def data():
    return load_data()

@st.cache_resource
def fitted():
    return train(data())

frame = data()
model, metrics, predictions = fitted()
forecast_tab, explore_tab, evaluation_tab = st.tabs(['Forecast replay', 'Explore demand', 'Model performance'])
with forecast_tab:
    st.write('Choose an observed hour in the held-out test period. Each forecast uses calendar information and demand observed 1, 24, and 168 hours earlier.')
    day = st.date_input('Forecast date', value=pd.Timestamp('2012-10-15').date(),
                       min_value=pd.Timestamp('2012-10-01').date(), max_value=pd.Timestamp('2012-12-31').date())
    hour = st.slider('Forecast hour', 0, 23, 8)
    timestamp = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    x, y = prepare(frame)
    if timestamp not in predictions.index:
        st.info('This hour or one of its required historical observations is missing. Choose another hour.')
    else:
        row = predictions.loc[timestamp]
        left, middle, right = st.columns(3)
        left.metric('Predicted rentals', f"{row['prediction']:.0f}")
        middle.metric('Observed rentals', f"{row['actual']:.0f}")
        right.metric('Weekly baseline', f"{row['weekly_baseline']:.0f}")
        st.line_chart(predictions.loc[timestamp - pd.Timedelta(hours=24):timestamp, ['actual', 'prediction']])
        st.write('Inputs available before the forecast hour')
        st.dataframe(x.loc[[timestamp], FEATURES], hide_index=True)
with explore_tab:
    st.subheader('Daily rental totals')
    st.line_chart(frame['cnt'].resample('D').sum())
    st.subheader('Typical hourly demand')
    profile = frame.assign(day_type=frame.index.dayofweek < 5).groupby(['hr', 'day_type'])['cnt'].mean().unstack()
    profile.columns = ['Weekend', 'Weekday']
    st.line_chart(profile)
    st.caption('Weekday/weekend groups do not exclude public holidays.')
with evaluation_tab:
    st.subheader(metrics['selected_model'])
    improvement = 100 * (1 - metrics['test_model']['MAE'] / metrics['test_weekly_baseline']['MAE'])
    st.metric('Test MAE (rentals per hour)', f"{metrics['test_model']['MAE']:.1f}")
    st.metric('MAE reduction versus weekly baseline', f'{improvement:.1f}%')
    st.json(metrics)
    st.write('Models were selected on July–September 2012, refitted through September, and evaluated on October–December. Actual preceding-hour demand updates the inputs at every forecast origin; this is not a multi-hour forecast.')
    st.write('Missing lag observations are excluded. Performance applies to this historical system and period. No prediction intervals or causal explanations are claimed.')
st.caption('Source: UCI Bike Sharing Dataset, Fanaee-T & Gama · DOI: 10.24432/C5W894 · CC BY 4.0')
