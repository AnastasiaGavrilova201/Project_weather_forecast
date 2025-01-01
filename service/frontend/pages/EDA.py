import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sktime.forecasting.naive import NaiveForecaster
from log import Logger
from plotly.subplots import make_subplots

logger = Logger(__name__).get_logger()

st.title("EDA")
st.sidebar.success("Просмотр EDA")

logger.info("Пользователь находится на странице с EDA-частью")
data = pd.read_csv('data/hourly_data_dec.csv')

data = data.drop(columns='Unnamed: 0')
st.info("Превью исторических данных о погоде:")
data_for_preview = data.copy()
data_for_preview = data_for_preview[[
    'dt', 'temp', 'pressure', 'humidity', 'wind_speed']]
data_for_preview.columns = [
    'Дата и время',
    'Температура (\u00B0C)',
    'Давление (мм)',
    'Влажность (%)',
    'Скорость ветра (м/c)']
st.dataframe(data_for_preview)


st.header("Общая динамика погодных данных")

df = data.copy()

df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
max_date = df['dt'].max()
last_week_data = df[(df['dt'] > max_date -
                     pd.Timedelta(days=5)) & (df['dt'] <= max_date)]
hourly_data = last_week_data[['dt', 'temp',
                              'pressure', 'humidity', 'wind_speed']]

last_data = df[df['dt'] == max_date]
weather_matching = {
    'Clouds': 'Облачно',
    'Rain': 'Дождь',
    'Clear': 'Ясно',
    'Snow': 'Снег',
}
most_frequent_weather = weather_matching[last_data['weather_main'][0]]


fig = make_subplots(
    rows=2, cols=4,
    column_widths=[0.25, 0.25, 0.25, 0.25],
    row_heights=[0.2, 0.8],
    specs=[
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        [{"type": "xy", "colspan": 4}, None, None, None]
    ],
    horizontal_spacing=0.05
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=last_data['temp'][0],
        title={
            "text": "Температура",
            "font": {
                "size": 12}},
        number={
            "suffix": "°C",
            "prefix": "+" if last_data['temp'][0] > 0 else None,
            "font": {
                "size": 32}},
        number_font_color=(
            '#87CEEB' if last_data['temp'][0] < 0 else '#FFB366'),
    ),
    row=1,
    col=1)

fig.add_trace(go.Indicator(
    mode="number",
    value=last_data['wind_speed'][0],
    title={"text": "Скорость ветра", "font": {"size": 12}},
    number={"suffix": " м/с", "font": {"size": 32}},
), row=1, col=2)

fig.add_trace(go.Indicator(
    mode="number",
    value=last_data['pressure'][0],
    title={"text": "Давление", "font": {"size": 12}},
    number={"suffix": " hPa", "font": {"size": 32}},
), row=1, col=3)

fig.add_trace(go.Indicator(
    mode="number",
    value=last_data['humidity'][0],
    title={"text": "Влажность", "font": {"size": 12}},
    number={"suffix": "%", "font": {"size": 32}},
), row=1, col=4)

fig.add_trace(
    go.Scatter(
        x=hourly_data['dt'],
        y=hourly_data['temp'],
        mode='lines',
        name='Температура',
        legendgroup="1",
        legendgrouptitle_text="Динамика"

    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=hourly_data['dt'],
        y=hourly_data['pressure'],
        mode='lines',
        name='Давление',
        legendgroup="1",
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=hourly_data['dt'],
        y=hourly_data['humidity'],
        mode='lines',
        name='Влажность',
        legendgroup="1",
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=hourly_data['dt'],
        y=hourly_data['wind_speed'],
        mode='lines',
        name='Скорость ветра',
        legendgroup="1",
    ),
    row=2, col=1
)

fig.update_xaxes(
    title_text="Динамика за последние 5 дней",
    title_standoff=35,
    row=2, col=1
)

fig.update_layout(
    #title="Погодная инфографика",
    template="plotly_dark",
    showlegend=True,
    legend=dict(
        groupclick="toggleitem",
        tracegroupgap=15
    ),
    margin=dict(t=100, b=100, l=80, r=80)
)

st.plotly_chart(fig)

data = data.iloc[:120, :]


def graph(column):
    forecaster = NaiveForecaster(window_length=24, strategy='mean')
    y = data[column]
    y.index = pd.date_range(
        start=min(
            data['dt'])[
            :16],
        end=max(
            data['dt'])[
            :16],
        freq="h").to_period()
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=range(1, 25))
    y_title = {
        'temp': '\u00B0C',
        'pressure': 'hPa',
        'wind_speed': 'м/с',
        'humidity': '%'}
    gragh_title = {
        'temp': 'температуры',
        'pressure': 'давления',
        'wind_speed': 'скорости ветра',
        'humidity': 'влажности'}
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
                              name='Наивный прогноз на следующие сутки'
                              )
                   )

    fig1.update_layout(title=f'Динамика {gragh_title[column]}',
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
