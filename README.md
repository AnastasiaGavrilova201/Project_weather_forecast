# Project_weather_forecast

**Проект №35**

**Тема:** 
Предсказание погоды для города/городов

**Куратор:** Васильев Николай @nnvasilevkk

**Участники (Команда 54):**  
- Гаврилова Анастасия Витальевна @kadaobed  
- Ковырков Денис Андреевич @denis_kovyrkov  
- Лысов Игорь Игоревич @lysov1  
- Панченко Василий Игоревич @panchenko_vi  
- Чуканов Тимофей Вячеславович @Timm27


## Описание проектa ##
  
### Цель проекта
Разработка и реализация сервиса для предсказания погоды в нескольких городах при помощи ML и TS на основе исторических данных о погоде, полученных через API.

### Результаты
- Сервис, предоставляющий предсказания погоды на основе классических моделей машинного обучения и моделей временных рядов для города или нескольких городов.
-	Документация, описывающая все этапы проекта, включая методы получения данных через API, предобработку, анализ, обучение классических моделей и моделей временных рядов, а также их оптимизацию.
-В зависимости от исходных данных можно предсказывать следующие погодные условия:
    *	Температура воздуха (максимальная, минимальная, средняя)
    *	Осадки (дождь, снег, количество осадков в мм)
    *	Влажность воздуха
    *	Атмосферное давление
    *	Скорость и направление ветра


### Дополнительные возможности
-	Интеграция с другими API для получения данных о климатических условиях или дополнительных метеорологических параметрах.
-	Визуализация данных и предсказаний на интерактивных графиках в микросервисе.

## Пользовательская инструкция

Приложение имеет 3 страницы:
1) Стартовая страница (вкладка Start)
2) Страница с EDA-частью (вкладка EDA)
3) Страница с ML-частью (вкладка ML)
   
Для переключения между страницами пользователь должен кликнуть на соответствующую вкладку
слева.

### Описаение вкладок 

1) Вкладка Start
   - Краткая навигация по приложению.
3) Вкладка EDA
   - Содержит графики временных рядов для следующих погодных данных: температура,
влажность, давление, скорость ветра за последние 7 дней (на основе доступных данных), а
также их прогноз на следующие сутки и некоторую описательную статистику.
3) Вкладка ML
   - Раздел «Загрузка данных»
     * Пользователь загружает последние данные о погоде в формате csv-файла в окно для загрузки и нажимает кнопку «Сохранить загруженные данные».
   - Раздел «Создание нового класса модели»
     * Пользователь вручную вводит id и гиперпараметры новой модели, которую он далее хочет
обучить, и нажимает кнопку «Создать новый класс для модели».
   - Раздел «Обучение модели»
     * В разделе описаны параметры и кривая обучения предобученной LSTM-модели.
   - Раздел «Загруженные модели»
     * При нажатии на кнопку «Показать все загруженные модели» приложение выводит список
всех загруженных и обученных ML-моделей.
   - Раздел «Установка активной модели»
     * Пользователь вручную вводит id модели, которая далее будет обучаться и по которой будет
строиться прогноз, и нажимает кнопку «Установить активную модель».
   - Раздел «Обучение активной модели»
     * Пользователь нажимает кнопку «Обучить активную модель».
   - Раздел «Прогноз активной модели»
     * Пользователь вводит дату и время, на которое строится прогноз, и горизонт
прогнозирования и нажимает кнопку «Показать прогноз температуры».
