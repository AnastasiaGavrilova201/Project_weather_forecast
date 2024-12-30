import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sktime.forecasting.naive import NaiveForecaster

st.title("EDA")
st.sidebar.success("Просмотр EDA")

data = pd.read_csv('data/Custom_location.csv')
data = data.iloc[-(5*24):, :]

st.info("Превью данных:")
st.dataframe(data)


def graph(column):
    forecaster = NaiveForecaster(window_length=24, strategy='mean')
    y = data[column]
    y.index = pd.date_range(start=min(data['dt_iso'])[:16], end=max(data['dt_iso'])[:16], freq="h").to_period()
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=range(1, 25))
    y_title = {'temp': u'\u00B0C', 'pressure': 'мм', 'wind_speed': 'м/с', 'humidity': '%'}
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y.index.to_timestamp(),
                              y=y,
                              mode='lines',
                              name='Историчные данные'
                              )
                   )
    fig1.add_trace(go.Scatter(x=y_pred.index.to_timestamp(),
                              y=y_pred,
                              mode='lines',
                              name='Прогноз'
                              )
                   )

    fig1.update_layout(#title='Прогноз температуры',
                       xaxis_title='Дата',
                       yaxis_title=y_title[column],
                       height=500,
                       width=1000
                       )

    st.plotly_chart(fig1)

st.header("Температура")
graph('temp')

st.header("Влажность")
graph('humidity')

st.header("Ветер")
graph('wind_speed')

st.header("Давление")
graph('pressure')


