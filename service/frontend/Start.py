import streamlit as st
from log import Logger

logger = Logger(__name__).get_logger()

st.set_page_config(page_title="This is a Multipage WebApp")
st.title("Прогнозирование погоды в Москве")
st.sidebar.success("Выберите страницу для просмотра")

st.info(
    'Вы находитесь на начальной странице приложения.\n\n'
    'Для просмотра анализа данных (EDA), нажмите вкладку "EDA" на панели слева.\n\n'
    'Для обучения ML-модели и прогнозирования погоды, нажмите вкладку "ML" на панели слева.'
)
logger.info("Пользователь находится на стартовой странице")
