import os

from lstm_src import Temp
from postgres import DatabaseManager
from utils import CsvToDatabase
from log import Logger

import tensorflow as tf  # pylint: disable=import-error

if len(tf.config.list_physical_devices('GPU')) == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = Logger(__name__).get_logger()


class Model:
    """
    A class representing a machine learning model for weather data.

    Attributes:
        table_nm (str): Name of the database table associated with the model.
        database (DatabaseManager): Instance of DatabaseManager for database operations.
        model (Temp): Instance of Temp representing the model's architecture and training logic.
        name (str): Name of the model.
        n_epochs (int): Number of epochs for training.
        desc (dict): Description of the model, including metadata and training status.
    """
    def __init__(self, table_nm, name, n_epochs=1):
        """
        Initializes the Model with a database table, model name, and training epochs.

        Args:
            table_nm (str): The name of the database table to use.
            name (str): The name of the model.
            n_epochs (int, optional): Number of epochs for training. Defaults to 10.
        """
        self.table_nm = table_nm
        self.database = DatabaseManager(self.table_nm)
        self.model = Temp(name, self.database, n_epochs)
        self.name = name
        self.n_epochs = n_epochs
        self.desc = {
            'name': self.name,
            'table_nm': self.table_nm,
            'n_epochs': self.n_epochs,
            'fitted': self.model.is_fitted()
        }
        logger.debug("inited model %s", self.name)

    def fit(self):
        """
        Trains the model using the associated data and updates its description.
        """
        self.model.fit()
        self.desc['fitted'] = self.model.is_fitted()
        logger.debug("finished fit model %s", self.name)

    def predict(self, start_time: str):
        """
        Makes predictions using the trained model.

        Args:
            start_time (str): The start time for predictions.

        Returns:
            Dataframe: The predictions made by the model. None if the model is not fitted.
        """
        if not self.model.is_fitted():
            logger.warning("call of 'predict' for not fitted model %s", self.name)
            return None
        return self.model.predict(start_time)


class API_Backend:
    """
    A backend interface to manage LSTM models and interact with them.

    Attributes:
        main_db_table_name (str): Name of the primary model's database table.
        main_model (Model): Instance of the primary model.
        second_model (Model, optional): Instance of a secondary model. Defaults to None.
        active_model (Model): The currently active model for operations.
    """
    def __init__(self):
        """
        Initializes the API backend with a primary model and prepares for operations.
        """
        self.main_db_table_name = 'test_realtime_6'
        self.main_model = Model(self.main_db_table_name, 'Main')
        self.second_model = None
        self.active_model = self.main_model
        logger.debug("Api backend initialized")

    def fit(self):
        """
        Fits the currently active model.
        """
        self.active_model.fit()

    def get_loaded_models(self):
        """
        Retrieves metadata for all loaded models.

        Returns:
            list: A list of dictionaries containing model descriptions.
        """
        if self.second_model is None:
            return [self.main_model.desc]
        return [self.main_model.desc, self.second_model.desc]

    def predict(self, start_time):
        """
        Makes predictions using the currently active model.

        Args:
            start_time (str): The start time for predictions.

        Returns:
            Dataframe: The predictions made by the active model.
        """
        return self.active_model.predict(start_time).to_json()

    def load_new_model(self, csv_path=None, table_nm='test_realtime_6', name='Second', n_epochs=5):
        """
        Loads a new model by uploading data and initializing the model.

        Args:
            csv_path (str): Path to the CSV file containing data in OWM format.
            table_nm (str): Name of the database table for the new model.
            name (str, optional): Name of the new model. Defaults to 'Second'.
            n_epochs (int, optional): Number of epochs for training the new model. Defaults to 10.
        """
        if csv_path:
            database = DatabaseManager(table_nm)
            if table_nm != self.main_model.table_nm:
                database.drop_table(table_nm)
                database.create_table(table_nm)
            loader = CsvToDatabase(database)
            loader.upload_csv_to_db(csv_path, table_nm, batch_size=2048)

        self.second_model = Model(table_nm, name, n_epochs)
        logger.debug("loaded new model '%s'", name)

    def set_active(self, name):
        """
        Sets the active model by name.

        Args:
            name (str): Name of the model to set as active.
        """
        if self.second_model.name == name:
            self.active_model = self.second_model
            logger.debug("changed active model to %s", self.second_model.name)
        else:
            self.active_model = self.main_model
            logger.debug("changed active model to %s", self.main_model.name)
