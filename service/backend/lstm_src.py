import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf  # pylint: disable=import-error
from tensorflow.keras.metrics import MeanAbsoluteError  # pylint: disable=import-error

from log import Logger

logger = Logger(__name__).get_logger()


class Temp:
    """
    A class to handle the training, transformation, and prediction processes for a weather forecasting model via LSTM.

    Attributes:
        name (str): Name of the model.
        database (DatabaseManager or None): Database manager for data operations.
        multivariate_data_config (dict): Configuration for multivariate data preparation with the following keys:
            - 'history_size' (int): Number of past time steps to consider for forecasting.
            - 'target_size' (int): Number of future time steps to forecast.
            - 'step' (int): Step size for data sampling.
        hyperparameters (dict): Dictionary containing hyperparameters for the model with the following keys:
            - 'BATCH_SIZE' (int): Batch size for training.
            - 'BUFFER_SIZE' (int): Buffer size for shuffling the dataset.
            - 'TRAIN_SPLIT' (int): Number of samples to use for the training split.
            - 'VAL_SIZE' (int): Number of samples to use for the validation split.
            - 'EVALUATION_INTERVAL' (int): Steps per epoch for evaluation.
            - 'EPOCHS' (int): Number of training epochs.
            - 'data_mean' (np.array): Array of mean values for data normalization.
            - 'data_std' (np.array): Array of standard deviation values for data normalization.
        datasets (dict): Dictionary containing datasets for training and validation with the following keys:
            - 'x_train' (np.array or None): Input features for training.
            - 'y_train' (np.array or None): Target labels for training.
            - 'x_val' (np.array or None): Input features for validation.
            - 'y_val' (np.array or None): Target labels for validation.
        model_filename (str): File path to save or load the trained model.
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
        self.database = database
        self.multivariate_data_config = {
            'history_size': 720,
            'target_size': 72,
            'step': 6
        }
        self.hyperparameters = {
            'BATCH_SIZE': 256,
            'BUFFER_SIZE': 10000,
            'TRAIN_SPLIT': 300000,
            'VAL_SIZE': 600000,
            'EVALUATION_INTERVAL': 200,
            'EPOCHS': n_epochs,
            'data_mean': np.array([744.41406611, 6.006443, 87.24904045, 224.61552684]),
            'data_std': np.array([216.24356615, 11.74853683, 112.89627056, 152.0149743])
        }
        self.datasets = {
            'x_train': None,
            'y_train': None,
            'x_val': None,
            'y_val': None
        }
        self.model_filename = f'./models/{self.name}.keras'
        if len(tf.config.list_physical_devices('GPU')) == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _multivariate_data(self, dataset, target, index_range, config):
        """
        Prepares multivariate data for training or validation.

        Args:
            dataset (np.array): The dataset array.
            target (np.array): Target values.
            index_range (tuple): A tuple containing the start and end indices for data slicing.
                             The start index is inclusive, and the end index is exclusive.
            config (dict): A dictionary containing the configuration with keys:
                - 'history_size' (int): Number of past steps to include.
                - 'target_size' (int): Number of future steps to forecast.
                - 'step' (int): Step size for sampling.
                - 'single_step' (bool, optional): Whether to forecast a single step. Defaults to False.

        Returns:
            tuple: Arrays of data and labels for the model.
        """
        start_index, end_index = index_range
        history_size = config['history_size']
        target_size = config['target_size']
        step = config['step']
        single_step = config.get('single_step', False)
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
            df['dt'] = df['dt'].apply(lambda row: str(row).split('+', maxsplit=1)[0])
            df['temp'] = df['temp'].astype(np.float64)
        else:
            df = pd.read_csv('./tmp/tes_osnova.csv')
        ss = pd.read_csv('./tmp/sunrise_sunset_2026.csv').iloc[:, 1:]

        ss['date'] = ss['date'] + ' 00:00:00'
        df = df.merge(ss, left_on='dt', right_on='date', how='left').drop('date', axis=1)
        df = df[['dt', 'diff_rise_set', 'temp', 'new_feature_1', 'new_feature_2']]
        df['dt'] = pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)

        if df.index.duplicated().any():
            logger.warning("Обнаружены дубликаты!")
            df = df[~df.index.duplicated(keep='first')]

        new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
        features = df.reindex(new_index)
        features = features.interpolate(method='linear')
        features.reset_index(inplace=True)
        features.rename(columns={'index': 'dt'}, inplace=True)
        features = features[features['dt'] < '2024-01-01 00:00:00'].dropna()
        total_samples = self.hyperparameters['TRAIN_SPLIT'] + self.hyperparameters['VAL_SIZE']
        features = features[-total_samples:].reset_index(drop=True)
        features = features.round(3)
        features.set_index('dt', inplace=True)

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Данные содержат NaN или бесконечные значения!")

        logger.info('TRAIN_SPLIT period: %(min)s - %(max)s', {
            'min': features[:self.hyperparameters['TRAIN_SPLIT']].index.min(),
            'max': features[:self.hyperparameters['TRAIN_SPLIT']].index.max()
        })

        dataset = features.values

        if np.any(self.hyperparameters['data_std'] == 0):
            logger.warning("Стандартное отклонение равно нулю!")

        dataset = (dataset-self.hyperparameters['data_mean']) / self.hyperparameters['data_std']

        self.datasets['x_train'], self.datasets['y_train'] = self._multivariate_data(
            dataset,
            dataset[:, 1],
            (0, self.hyperparameters['TRAIN_SPLIT']),
            self.multivariate_data_config
        )

        self.datasets['x_val'], self.datasets['y_val'] = self._multivariate_data(
            dataset,
            dataset[:, 1],
            (self.hyperparameters['TRAIN_SPLIT'], None),
            self.multivariate_data_config
        )

        logger.debug('x_train shape: %s', self.datasets['x_train'].shape)
        logger.debug('y_train shape: %s', self.datasets['y_train'].shape)

        logger.debug('x_val shape: %s', self.datasets['x_val'].shape)
        logger.debug('y_val shape: %s', self.datasets['y_val'].shape)

    def _fit(self):
        """
        Trains the model using the prepared dataset and saves the trained model to disk.
        """
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./report', exist_ok=True)

        train_data = tf.data.Dataset.from_tensor_slices(
            (self.datasets['x_train'],
             self.datasets['y_train'])).cache().shuffle(
                 self.hyperparameters['BUFFER_SIZE']).batch(
                 self.hyperparameters['BATCH_SIZE']).repeat()

        val_data = tf.data.Dataset.from_tensor_slices(
            (self.datasets['x_val'],
             self.datasets['y_val'])).batch(self.hyperparameters['BATCH_SIZE']).repeat()

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=self.datasets['x_train'].shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(72))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae', metrics=[MeanAbsoluteError()])

        history = model.fit(
            train_data,
            epochs=self.hyperparameters['EPOCHS'],
            steps_per_epoch=self.hyperparameters['EVALUATION_INTERVAL'],
            validation_data=val_data,
            validation_steps=50
        )

        tf.keras.models.save_model(model, self.model_filename)

        history_df = pd.DataFrame(history.history)
        history_df['date'] = datetime.now()
        history_df['loading_id'] = history_df['date'].apply(lambda row: row.timestamp())

        if os.path.exists('./report/lstm_temp_history.csv'):
            history_df.to_csv('./report/lstm_temp_history.csv', mode='a', header=False, index=False)
        else:
            history_df.to_csv('./report/lstm_temp_history.csv', index=False)

    def _multivariate_data2(self, dataset, target, config):
        """
        Prepares multivariate data for prediction.

        Args:
            dataset (np.array): The dataset array.
            target (np.array): Target values.
            config (dict): A dictionary containing the configuration with keys:
                - 'history_size' (int): Number of past steps to include.
                - 'target_size' (int): Number of future steps to forecast.
                - 'step' (int): Step size for sampling.
                - 'single_step' (bool, optional): Whether to forecast a single step. Defaults to False.

        Returns:
            tuple: Arrays of data and labels for prediction.
        """
        history_size = config['history_size']
        target_size = config['target_size']
        step = config['step']
        single_step = config.get('single_step', False)

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

    def _get_raw_data(self):
        """
        Fetches raw data from the database or a CSV file.

        Returns:
            pd.DataFrame: The raw dataset.
        """
        if self.database:
            logger.debug('using data from database')
            df = self.database.select(header=['dt', 'temp'], additional_options="where dt > '2024-01-01' order by dt")
            df['temp'] = df['temp'].astype(np.float64)
        else:
            df = pd.read_csv('./tmp/test_realtime_2.csv')
        logger.info('Последняя дата в датафрейме : %s', df["dt"].max())
        df = df.dropna(axis=1)
        df['dt'] = df['dt'].apply(lambda row: str(row).split('+', maxsplit=1)[0])
        return df

    def _get_history_data(self, start, history_samples, df):
        """
        Fetches historical data from the dataset.

        Args:
            start (str): The start date for fetching historical data.
            history_samples (int): The number of historical samples to retrieve.
            df (pd.DataFrame): The dataset.

        Returns:
            pd.DataFrame: Historical data.
        """
        df['dt'] = pd.to_datetime(df['dt'])
        df = df[df['dt'] < start]
        return df.tail(history_samples)

    def _prepare_data_for_predict(self, start, df):
        """
        Prepares data for prediction, including feature engineering and normalization.

        Args:
            start (str): The start date for prediction.
            df (pd.DataFrame): The dataset.

        Returns:
            tuple: Prepared x_val and y_val data for prediction.
        """
        ss = pd.read_csv('./tmp/sunrise_sunset_2026.csv').iloc[:, 1:]
        ss['date'] = ss['date'] + ' 00:00:00'
        ss2 = ss[ss['date'] >= start][['date', 'diff_rise_set', 'new_feature_1', 'new_feature_2']].head(3).copy()
        df = df.merge(ss, left_on='dt', right_on='date', how='left').drop('date', axis=1)
        df = df[['dt', 'diff_rise_set', 'temp', 'new_feature_1', 'new_feature_2']]
        df['dt'] = pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)

        new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
        features = df.reindex(new_index)
        features = features.interpolate(method='linear')
        features.reset_index(inplace=True)
        features.rename(columns={'index': 'dt'}, inplace=True)
        features = features[features['dt'] < start].dropna()
        features.set_index('dt', inplace=True)

        last_12_values = features['temp'].tail(72).tolist()
        ss2['date'] = pd.to_datetime(ss2['date'])
        ss2.set_index('date', inplace=True)
        new_index = pd.date_range(start=ss2.index.min(), end=ss2.index.max(), freq='10T')
        features2 = ss2.reindex(new_index).interpolate(method='time')
        features2 = features2.head(72)
        features2.insert(1, 'temp', last_12_values)

        features = pd.concat([features, features2])

        dataset = features.values
        dataset = (dataset-self.hyperparameters['data_mean']) / self.hyperparameters['data_std']

        x_val, y_val = self._multivariate_data2(dataset,
                                                dataset[:, 1],
                                                self.multivariate_data_config)
        return x_val, y_val

    def _predict(self, start, x_val, y_val):
        """
        Generates predictions using the trained model.

        Args:
            start (str): The start date for predictions.
            x_val (np.array): Input data for prediction.
            y_val (np.array): Target values for validation.

        Returns:
            tuple: Date range and predicted values.
        """
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.hyperparameters['BATCH_SIZE'])
        model = tf.keras.models.load_model(self.model_filename)

        predictions = model.predict(val_data)
        predictions = (
            predictions * self.hyperparameters['data_std'][1]
            + self.hyperparameters['data_mean'][1]
        ).flatten()

        num_groups = len(predictions) // 6
        averages = []
        for i in range(num_groups):
            group = predictions[(i * 6):(i + 1) * 6]
            average = np.round(np.mean(group), 2)
            averages.append(average)

        date_range = pd.date_range(start=start, end=pd.to_datetime(start) + pd.Timedelta(hours=11), freq='H')
        return date_range, averages

    def predict(self, start, history_samples=0):
        """
        Generates forecasts based on the given start date.

        Args:
            start (str): The start date for forecasting.
            history_samples (int, optional): Number of historical samples to include in the output. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing forecasted values and their corresponding timestamps.
        """
        raw_data = self._get_raw_data()
        if history_samples > 0:
            history_data = self._get_history_data(start, history_samples, raw_data.copy())
        x_val, y_val = self._prepare_data_for_predict(start, raw_data)
        date_range, averages = self._predict(start, x_val, y_val)
        forecast = pd.DataFrame({'dt': date_range, 'temp': averages})
        if history_samples > 0:
            print(history_samples)
            return pd.concat([history_data, forecast], ignore_index=True, axis=0)
        return forecast

    def is_fitted(self):
        """
        Checks whether the model has been fitted and saved to disk.

        Returns:
            bool: True if the model file exists, False otherwise.
        """
        return os.path.exists(self.model_filename)

    def fit(self):
        """
        Transforms the data and trains the model.
        """
        self._transform()
        self._fit()
