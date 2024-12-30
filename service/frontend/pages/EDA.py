import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sktime.forecasting.naive import NaiveForecaster
from log import Logger

logger = Logger(__name__).get_logger()

st.title("EDA")
st.sidebar.success("Просмотр EDA")

logger.info("Пользователь находится на странице с EDA-частью")
data = pd.read_csv('data/hourly_data.csv')
#data = data.iloc[-(5*24):, :]
data = data.drop(columns = 'Unnamed: 0')
st.info("Превью исторических данных о погоде:")
data_for_preview = data.copy()
data_for_preview.columns = ['Дата и время', 'Температура (\u00B0C)', 'Давление (мм)', 'Влажность (%)', 'Скорость ветра (м/c)']
st.dataframe(data_for_preview)

def graph(column):
    forecaster = NaiveForecaster(window_length=24, strategy='mean')
    y = data[column]
    y.index = pd.date_range(start=min(data['dt'])[:16], end=max(data['dt'])[:16], freq="h").to_period()
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=range(1, 25))
    y_title = {'temp': '\u00B0C', 'pressure': 'мм', 'wind_speed': 'м/с', 'humidity': '%'}
    gragh_title = {'temp': 'температуры', 'pressure': 'давления', 'wind_speed': 'скорости ветра', 'humidity': 'влажности'}
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x = y.index.to_timestamp(),
                              y = y,
                              mode = 'lines',
                              name = 'Историчные данные'
                              )
                   )
    fig1.add_trace(go.Scatter(x = y_pred.index.to_timestamp(),
                              y = y_pred,
                              mode = 'lines',
                              name = 'Наивный прогноз на следующие сутки'
                              )
                   )

    fig1.update_layout(title = f'Динамика {gragh_title[column]}',
                       xaxis_title = 'Дата',
                       yaxis_title = y_title[column],
                       height = 500,
                       width = 1000
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