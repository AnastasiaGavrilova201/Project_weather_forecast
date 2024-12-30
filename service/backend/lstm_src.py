import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.metrics import MeanAbsoluteError

from log import Logger

logger = Logger(__name__).get_logger()

class Temp:
    """
    A class to handle the training, transformation, and prediction processes for a weather forecasting model via LSTM.

    Attributes:
        name (str): Name of the model.
        past_history (int): Number of past time steps used for forecasting.
        future_target (int): Number of future time steps to forecast.
        STEP (int): Step size for data sampling.
        BATCH_SIZE (int): Batch size for training.
        BUFFER_SIZE (int): Buffer size for shuffling the dataset.
        TRAIN_SPLIT (int): Number of samples for training split.
        VAL_SIZE (int): Number of samples for validation split.
        EVALUATION_INTERVAL (int): Steps per epoch for evaluation.
        EPOCHS (int): Number of training epochs.
        data_mean (np.array): Mean values for data normalization.
        data_std (np.array): Standard deviation values for data normalization.
        database (DatabaseManager or None): Database manager for data operations.
    """
    def __init__(self, model_name, database=None, n_epochs=10):
        """
        Initializes the Temp class with model configuration and data processing parameters.

        Args:
            model_name (str): Name of the model.
            database (DatabaseManager, optional): Instance of DatabaseManager for database operations. Defaults to None.
            n_epochs (int, optional): Number of epochs for training. Defaults to 10.
        """
        self.name = model_name
        self.past_history = 720
        self.future_target = 72
        self.STEP = 6
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 10000
        self.TRAIN_SPLIT = 300000
        self.VAL_SIZE = 600000
        self.EVALUATION_INTERVAL = 200
        self.EPOCHS = n_epochs
        self.data_mean = np.array([744.41406611, 6.006443, 87.24904045, 224.61552684])
        self.data_std  = np.array([216.24356615, 11.74853683, 112.89627056, 152.0149743])
        self.database = database
        if len(tf.config.list_physical_devices('GPU')) == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        

    def _multivariate_data(self, dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        """
        Prepares multivariate data for training or validation.

        Args:
            dataset (np.array): The dataset array.
            target (np.array): Target values.
            start_index (int): Starting index for data slicing.
            end_index (int or None): Ending index for data slicing. Defaults to None for the full range.
            history_size (int): Number of past steps to include.
            target_size (int): Number of future steps to forecast.
            step (int): Step size for sampling.
            single_step (bool, optional): Whether to forecast a single step. Defaults to False.

        Returns:
            tuple: Arrays of data and labels for the model.
        """
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    def _transform(self):
        """
        Prepares and normalizes the dataset for training and validation.
        """
        if self.database:
            df = self.database.select(header=['dt', 'temp'], additional_options="where dt < '2024-01-01' order by dt")
            df['dt'] = df['dt'].apply(lambda row: str(row).split('+')[0])
            df['temp'] = df['temp'].astype(np.float64)
        else:
            df=pd.read_csv('./tmp/tes_osnova.csv')
        ss=pd.read_csv('./tmp/sunrise_sunset_2026.csv').iloc[:,1:]

        ss['date'] = ss['date'] + ' 00:00:00'
        df = df.merge(ss, left_on='dt', right_on='date', how='left').drop('date',axis=1)
        df = df[['dt','diff_rise_set','temp','new_feature_1','new_feature_2']]
        df['dt']=pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)

        if df.index.duplicated().any():
            logger.warning("Обнаружены дубликаты!")
            df = df[~df.index.duplicated(keep='first')]

        new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
        features = df.reindex(new_index)
        features = features.interpolate(method='linear')
        features.reset_index(inplace=True)
        features.rename(columns={'index': 'dt'}, inplace=True)
        features_base = features.copy()
        features=features[features['dt']<'2024-01-01 00:00:00'].dropna()
        features=features[-(self.TRAIN_SPLIT + self.VAL_SIZE):].reset_index(drop=True)
        features=features.round(3)
        features.set_index('dt', inplace=True)

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Данные содержат NaN или бесконечные значения!")

        logger.info(f'TRAIN_SPLIT period : {features[:self.TRAIN_SPLIT].index.min()} - {features[:self.TRAIN_SPLIT].index.max()}')

        dataset = features.values

        if np.any(self.data_std == 0):
            logger.warning("Стандартное отклонение равно нулю!")

        dataset = (dataset-self.data_mean) / self.data_std

        self.x_train, self.y_train = self._multivariate_data(dataset,
                                                                   dataset[:, 1],
                                                                   0,
                                                                   self.TRAIN_SPLIT,
                                                                   self.past_history,
                                                                   self.future_target,
                                                                   self.STEP)

        self.x_val, self.y_val = self._multivariate_data(dataset,
                                                               dataset[:, 1],
                                                               self.TRAIN_SPLIT,
                                                               None,
                                                               self.past_history,
                                                               self.future_target,
                                                               self.STEP)

        logger.debug(f'x_train shape: {self.x_train.shape}')
        logger.debug(f'y_train shape: {self.y_train.shape}')

        logger.debug(f'x_val shape: {self.x_val.shape}')
        logger.debug(f'y_val shape : {self.y_val.shape}')

    def _fit(self):
        """
        Trains the model using the prepared dataset and saves the trained model to disk.
        """
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./report', exist_ok=True)

        train_data = tf.data.Dataset.from_tensor_slices(
            (self.x_train,
             self.y_train)).cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()

        val_data = tf.data.Dataset.from_tensor_slices(
            (self.x_val,
             self.y_val)).batch(self.BATCH_SIZE).repeat()

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=self.x_train.shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(72))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae', metrics=[MeanAbsoluteError()])

        history = model.fit(train_data,
                    epochs=self.EPOCHS,
                    steps_per_epoch=self.EVALUATION_INTERVAL,
                    validation_data=val_data,
                    validation_steps=50)

        # model.save('./models/lstm_temp.h5')
        tf.keras.models.save_model(model, f'./models/{self.name}.keras')

        history_df = pd.DataFrame(history.history)
        history_df['date'] = datetime.now()
        history_df['loading_id'] = history_df['date'].apply(lambda row: row.timestamp())

        if os.path.exists('./report/lstm_temp_history.csv'):
            history_df.to_csv('./report/lstm_temp_history.csv', mode='a', header=False, index=False)
        else:
            history_df.to_csv('./report/lstm_temp_history.csv', index=False)

    def _multivariate_data2(self, dataset, target, history_size, target_size, step, single_step=False):
        """
        Prepares multivariate data for prediction.

        Args:
            dataset (np.array): The dataset array.
            target (np.array): Target values.
            history_size (int): Number of past steps to include.
            target_size (int): Number of future steps to forecast.
            step (int): Step size for sampling.
            single_step (bool, optional): Whether to forecast a single step. Defaults to False.

        Returns:
            tuple: Arrays of data and labels for prediction.
        """
        data = []
        labels = []

        end_index = len(dataset) - target_size
        start_index = end_index - history_size
        if start_index < 0:
            raise ValueError("History size is too large for the given dataset.")
        indices = range(start_index, end_index, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[end_index + target_size])
        else:
            labels.append(target[end_index:end_index + target_size])
        return np.array(data), np.array(labels)

    def predict(self, start):
        """
        Generates forecasts based on the given start date.

        Args:
            start (str): The start date for forecasting.

        Returns:
            pd.DataFrame: A DataFrame containing forecasted values and their corresponding timestamps.
        """
        if self.database:
            logger.debug('using data from database')
            df = self.database.select(header=['dt', 'temp'], additional_options="where dt > '2024-01-01' order by dt")
            df['temp'] = df['temp'].astype(np.float64)
        else:
            df=pd.read_csv('./tmp/test_realtime_2.csv')

        logger.info(f'Последняя дата в датафрейме : {df["dt"].max()}')
        df=df.dropna(axis=1)
        df['dt'] = df['dt'].apply(lambda row: str(row).split('+')[0])

        ss=pd.read_csv('./tmp/sunrise_sunset_2026.csv').iloc[:,1:]
        ss['date'] = ss['date'] + ' 00:00:00'
        ss2 = ss[ss['date'] >= start][['date','diff_rise_set', 'new_feature_1','new_feature_2']].head(3).copy()
        df = df.merge(ss, left_on='dt', right_on='date', how='left').drop('date',axis=1)
        df = df[['dt','diff_rise_set','temp','new_feature_1','new_feature_2']]
        df['dt']=pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)

        new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
        features = df.reindex(new_index)
        features = features.interpolate(method='linear')
        features.reset_index(inplace=True)
        features.rename(columns={'index': 'dt'}, inplace=True)
        features=features[features['dt'] < start].dropna()
        features.set_index('dt', inplace=True)

        last_12_values = features['temp'].tail(72).tolist()
        ss2['date'] = pd.to_datetime(ss2['date'])
        ss2.set_index('date', inplace=True)
        new_index = pd.date_range(start=ss2.index.min(), end=ss2.index.max(), freq='10T')
        features2 = ss2.reindex(new_index).interpolate(method='time')
        features2 = features2.head(72)
        features2.insert(1, 'temp', last_12_values)

        features = pd.concat([features,features2])

        dataset = features.values
        dataset = (dataset-self.data_mean) / self.data_std

        x_val, y_val = self._multivariate_data2(dataset,
                                                 dataset[:, 1],
                                                 self.past_history,
                                                 self.future_target,
                                                 self.STEP)

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)

        # model = tf.keras.models.load_model('lstm_temp.h5')
        model = tf.keras.models.load_model(f'./models/{self.name}.keras')

        predictions = model.predict(val_data)
        predictions = (predictions * self.data_std[1]  + self.data_mean[1]).flatten()

        num_groups = len(predictions) // 6
        averages = []
        for i in range(num_groups):
            group = predictions[i * 6 : (i + 1) * 6]
            average = np.round(np.mean(group),2)
            averages.append(average)

        date_range = pd.date_range(start=start, end=pd.to_datetime(start) + pd.Timedelta(hours=11), freq='H')
        forecast = pd.DataFrame({'date':date_range, 'forecast':averages})
        return forecast

    def fit(self):
        """
        Transforms the data and trains the model.
        """
        self._transform()
        self._fit()
