# Этапы проекта
1.	Получение данных через API
    -	Задача: Спроектировать и реализовать систему для автоматического сбора данных о погоде из города/городов с использованием API (например, Gismeteo, OpenWeatherMap).
    -	Подзадачи:
        *	Ознакомление с документацией API для получения исторических и текущих данных о погоде.
        *	Реализация скрипта для периодического запроса и сбора данных о погоде (температура, влажность, давление, скорость ветра и т.д.).
        *	Сохранение данных в формате CSV или базы данных (например, SQLite, PostgreSQL).
2.	Дата анализ данных
    -	Задача: Проанализировать собранные данные, чтобы понять их структуру, выявить закономерности и аномалии.
    -	Подзадачи:
        *	Исследование временных рядов данных для каждого города.
        *	Анализ сезонности, трендов и цикличности в данных о погоде.
        *	Выявление и обработка выбросов и пропусков в данных.
3.	Предобработка данных
    -	Задача: Привести данные к формату, пригодному для обучения моделей машинного обучения.
    -	Подзадачи:
        *	Обработка временных рядов: устранение пропусков, сглаживание данных.
        *	Преобразование данных в формат, удобный для моделирования временных рядов.
        *	Разделение данных на тренировочную и тестовую выборки с учетом временной структуры данных.
4.	Обучение классических моделей машинного обучения
    -	Задача: Построить и обучить классические модели машинного обучения для предсказания погоды на основе собранных данных.
    -	Подзадачи:
        *	Выбор классических моделей (например, линейная регрессия, случайный лес, градиентный бустинг).
        *	Обучение моделей на тренировочных данных.
        *	Оценка качества предсказаний на тестовых данных (метрики, например, MSE, MAE).
        *	Анализ ошибок и корректировка моделей.
5.	Feature engineering
    -	Задача: Улучшить качество классических моделей за счет создания новых признаков и их оптимизации.
    -	Подзадачи:
        *	Создание дополнительных признаков (например, лаги, взаимодействие признаков, сезонные компоненты).
        *	Отбор наиболее значимых признаков с учетом временной зависимости данных.
        *	Реобучение классических моделей на новом наборе признаков и сравнение результатов.
        *	Подбор гиперпараметров для оптимизации моделей (например, Grid Search, Random Search).
6.	Использование моделей временных рядов (Time Series)
    -	Задача: Исследовать возможность применения моделей временных рядов для повышения точности предсказания погоды.
    -	Подзадачи:
        *	Выбор моделей временных рядов (например, ARIMA, Prophet, LSTM).
        *	Обучение моделей на исторических данных.
        *	Оценка и сравнение производительности моделей временных рядов с классическими моделями.
        *	Анализ преимуществ и недостатков использования моделей временных рядов.
7.	Реализация микросервиса
    -	Задача: Разработать микросервис, предоставляющий доступ к модели предсказания погоды через API или пользовательский интерфейс.
    -	Подзадачи:
        *	Выбор платформы для реализации (например, Telegram Bot, Streamlit).
        *	Разработка интерфейса для взаимодействия с пользователем (например, получение данных о городе, предсказание погоды).
        *	Развертывание микросервиса на сервере или в облаке.
