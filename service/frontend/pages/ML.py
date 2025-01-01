import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sktime.forecasting.naive import NaiveForecaster
from log import Logger
import httpx
import json
from plotly.subplots import make_subplots

logger = Logger(__name__).get_logger()

logger.info("Пользователь находится на странице с ML-частью")
st.title("ML")
st.sidebar.success("Просмотр ML-части")

st.markdown('### Загрузка csv-файла')
uploaded_file = st.file_uploader("Выберите CSV-файл c данными", type=["csv"])
if uploaded_file is not None:
    upload_data = pd.read_csv(uploaded_file)
    st.info("Превью загруженных данных:")
    st.dataframe(upload_data)
    if st.button("Сохранить загруженный файл"):
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        files = {'file': (uploaded_file.name, file_content, 'csv')}
        response = httpx.post("http://localhost:8000/upload_csv", files=files)
        if response.status_code == 200:
            st.success("Ваш CSV-файл сохранен.")
            logger.info("Пользователь загрузил csv-файл")
        else:
            st.error(response.text)
            logger.error(response.text)
else:
    st.info("Пожалуйста, загрузите не пустой CSV-файл.")

st.markdown('### Создание нового класса модели')
model_new = st.text_input("Введите id модели", key="model_new")
st.markdown('##### Введите гиперпараметры новой модели')
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
                         value=10,
                         min_value=1)
if st.button("Создать новый класс для модели"):
    if model_new == '':
        st.error('Введите id модели')
        logger.warning("Нет id модели для создания нового класса")
    else:
        params = {
            # "csv_path": file_name,
            # "table_nm": model_new,
            "model_name": model_new,
            "n_epochs": epochs
        }
        response = httpx.post(
            "http://localhost:8000/load_new_model",
            json=params)
        if response.status_code == 200:
            st.success(f'Создан новый класс для модели {model_new}')
            logger.info("Пользователь создал новый класс модели")
        else:
            st.error(response.text)
            logger.error(response.text)


st.header("Предобученная модель")
st.markdown('##### Параметры LSTM-модели')
params = {'Параметр':
          ['past_history', 'step', 'batch_size',
           'buffer_size', 'train_split',
           'evaluation_interval', 'epochs'],
          'Значение': [720, 6, 256, 10000, 300000, 200, 10]}

st.dataframe(pd.DataFrame(params))

lstm_hist = pd.read_csv('data/lstm_temp_history.csv')
fig = go.Figure()
fig.add_trace(go.Scatter(x=lstm_hist['loss'].index + 1,
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
fig.update_layout(title='Кривая обучения',
                  xaxis_title='Эпоха',
                  yaxis_title='Потери (MAE)',
                  height=500,
                  width=1000
                  )
st.plotly_chart(fig)

st.markdown('### Загруженные модели')
if st.button("Показать все загруженные модели"):
    logger.info("Пользователь вывел все загруженные модели")
    response = httpx.get("http://localhost:8000/models")
    if response.status_code == 200:
        if (pd.DataFrame(response.json())).shape[0] == 0:
            st.info('Нет загруженных моделей')
        else:
            models_table = pd.DataFrame(response.json())
            st.success('id моделей:')
            st.dataframe(models_table)
    else:
        st.error(response.text)
        logger.error(response.text)


st.markdown('### Установка активной модели')
active_model_id = st.text_input("Введите id модели", key="active_model_id")
if st.button("Установить активную модель"):
    if active_model_id == '':
        st.error('Введите id модели')
        logger.warning("Нет id активной модели")
    else:
        params = {
            "model_name": active_model_id
        }
        response = httpx.post("http://localhost:8000/set_model", json=params)
        if response.status_code == 200:
            st.success(f'Установлена активная модель {active_model_id}')
            logger.info(response.text)
        else:
            st.error(response.text)
            logger.error(response.text)

st.markdown('### Обучение активной модели')

#model_id_fit = st.text_input("Введите id модели", key = "model_id_fit")

if st.button("Обучить активную модель"):
    response = httpx.post("http://localhost:8000/fit", timeout=None)
    if response.status_code == 200:
        st.success(f'Модель {active_model_id} обучена')
        logger.info("Пользователь обучил активную модель")
    else:
        st.error(response.text)
        logger.error(response.text)


st.markdown('### Прогноз активной модели')

date_time_forecast = st.text_input(
    "Введите начальную дату и время прогноза в формате YYYY-MM-DD HH:MM:SS",
    key="date_time_forecast")
options = {'3 часа': 3, '6 часов': 6, '9 часов': 9, '12 часов': 12}
forecast_horizon = st.selectbox("Прогноз на:", options.keys())

# data = pd.read_csv('data/hourly_data.csv')
# data = data[['temp', 'dt']]
# data = data.iloc[-24:, :]
if st.button("Показать прогноз температуры"):
    if date_time_forecast == '':
        st.error('Дата и время прогноза не указаны')
        logger.warning("Нет даты и времени для предсказания")
    else:
        params = {"start_time": date_time_forecast}
        response = httpx.post("http://localhost:8000/predict", json=params)
        if response.status_code == 200:
            predictions_str = response.json()['predictions']
            predictions_dict = json.loads(predictions_str)
# Преобразуем в DataFrame
            df = pd.DataFrame(predictions_dict)
# Преобразуем колонку 'dt' из миллисекунд в дату
            df['dt'] = pd.to_datetime(df['dt'], unit='ms')
            hourly_data = df.iloc[:-12, :]
            df_forescast = df.iloc[-12:, :]
            if options[forecast_horizon] == 12:
                st.dataframe(df.iloc[-12:].reset_index(drop=True))
            else:
                st.dataframe(
                    df.iloc[-12:(-12 + options[forecast_horizon])].reset_index(drop=True))
            logger.info(response.text)
            fig1 = make_subplots(
                rows=1, cols=4,
                column_widths=[0.25, 0.25, 0.25, 0.25],
                row_heights=[1.0],
                specs=[
                    [{"type": "xy", "colspan": 4}, None, None, None]
                ],
                horizontal_spacing=0.05
            )

            fig1.add_trace(
                go.Scatter(
                    x=hourly_data['dt'],
                    y=hourly_data['temp'],
                    mode='lines',
                    name='Температура',
                    legendgroup="2",
                    legendgrouptitle_text="Прогноз"
                ),
                row=1, col=1
            )
            fig1.add_trace(
                go.Scatter(
                    x=df_forescast['dt'],
                    y=df_forescast['temp'],
                    mode='lines',
                    name='Температура (прогноз)',
                    legendgroup="2",
                    legendgrouptitle_text="Прогноз"
                ),
                row=1, col=1
            )

            fig1.update_xaxes(
                title_text="Результат прогноза",
                title_standoff=35,
                row=1, col=1
            )

            fig1.update_layout(
                #title="Погодная инфографика",
                template="plotly_dark",
                showlegend=True,
                legend=dict(
                    groupclick="toggleitem",
                    tracegroupgap=15
                ),
                margin=dict(t=100, b=100, l=80, r=80)
            )

            st.plotly_chart(fig1)
        else:
            st.error(response.text)
            logger.error(response.text)
