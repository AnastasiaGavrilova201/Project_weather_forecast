import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sktime.forecasting.naive import NaiveForecaster
from log import Logger

logger = Logger(__name__).get_logger()

logger.info("Пользователь находится на странице с ML-частью")
st.title("ML")
st.sidebar.success("Просмотр ML-части")

uploaded_file = st.file_uploader("Выберите CSV-файл c данными о погоде", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.info("Превью данных:")
    st.dataframe(data)
else:
    st.info("Пожалуйста, загрузите не пустой CSV-файл.")

st.header("Обучение модели")

st.markdown('##### Параметры модели')

params = {'Параметр':
              ['past_history', 'step', 'batch_size',
               'buffer_size','train_split',
               'evaluation_interval','epochs'],
          'Значение' : [720, 6, 256, 10000, 300000, 200, 10]}

st.dataframe(pd.DataFrame(params))

lstm_hist = pd.read_csv('data/lstm_temp_history.csv')

fig = go.Figure()
fig.add_trace(go.Scatter(x=lstm_hist['loss'].index+1,
                         y=lstm_hist['loss'],
                         mode='lines',
                         name='Тренировочные данные'
                         )
              )
fig.add_trace(go.Scatter(x=lstm_hist['val_loss'].index + 1,
                         y=lstm_hist['val_loss'],
                         mode='lines',
                         name='Валидационные данные'
                         )
              )
fig.update_layout(title='Динамика потерь во время обучения модели (MAE)',
                  xaxis_title='Эпоха',
                  yaxis_title='Потери',
                  height=500,
                  width=1000
                  )

st.plotly_chart(fig)

st.header("Обучение новой модели")



st.markdown('### Обучение модели')

model_id_fit = st.text_input("Введите id новой модели")
st.markdown('##### Введите гиперпараметры модели')
past_history = st.number_input("Укажите past_history модели",
                               value=720)
step = st.number_input("Укажите step модели",
                       value=6,
                       min_value=1)
batch_size = st.number_input("Укажите batch_size модели",
                             value=256,
                             min_value=1)
buffer_size = st.number_input("Укажите buffer_size модели",
                              value=10000)
train_split = st.number_input("Укажите train_split модели",
                              value=300000)
evaluation_interval = st.number_input("Укажите evaluation_interval модели",
                                      value=200)
epochs = st.number_input("Укажите epochs модели",
                         value=10)

if st.button("Обучить новую модель"):
    if model_id_fit == '':
        st.error('Введите id модели')
        logger.warning("Нет id модели для обучения")
    else:
        st.success(f'Модель {model_id_fit} обучена')
        logger.info("Пользователь обучил новую модель")

if st.button("Показать все загруженные модели"):
    logger.info("Пользователь вывел все загруженные модели")
    response = {'id': ['model1']}
    models_table = pd.DataFrame(response)
    if len(response.keys()) == 0:
        st.error('Нет загруженных моделей')
    else:
        st.success('id моделей:')
        st.dataframe(models_table)

st.markdown('### Прогноз модели')

model_id_forecast = st.text_input("Введите id модели")
options = {'3 часа': 3, '6 часов': 6, '9 часов': 9, '12 часов':12}
forecast_horizon = st.selectbox("Прогноз на:", options.keys())

data = pd.read_csv('data/Custom_location.csv')
data = data[['temp', 'dt_iso']]
data = data.iloc[-24:, :]
if st.button("Показать прогноз температуры"):
    if model_id_forecast == '':
        st.error('Введите id модели')
        logger.warning("Нет id модели для предсказания")
    else:
        forecaster = NaiveForecaster(window_length=24, strategy='mean')
        y = data['temp']
        y.index = pd.date_range(start = min(data['dt_iso'])[:16], end = max(data['dt_iso'])[:16], freq="h").to_period()
        forecaster.fit(y)
        y_pred = forecaster.predict(fh = range(1, options[forecast_horizon] + 1))
        st.markdown(f'#### Прогноз на {forecast_horizon} в табличном виде (\u00B0C)')
        predict_temp_table = pd.DataFrame(y_pred)
        predict_temp_table.columns = ['Температура']
        st.dataframe(predict_temp_table)
        st.markdown(f'#### Прогноз на {forecast_horizon} в виде графика')
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
                                  name = 'Прогноз'
                                  )
                       )

        fig1.update_layout(title = 'Прогноз температуры',
                           xaxis_title = 'Дата',
                           yaxis_title = u'\u00B0C',
                           height = 500,
                           width = 1000
                           )
        st.plotly_chart(fig1)
        logger.info("Пользователь рассчитал прогноз погоды")