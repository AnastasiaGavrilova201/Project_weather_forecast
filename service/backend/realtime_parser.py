import os
from datetime import datetime
import time
import traceback
from copy import deepcopy
from threading import Thread
import requests
from requests.exceptions import RequestException

from postgres import DatabaseManager
from log import Logger

logger = Logger(__name__).get_logger()


class Api:
    """
    Handles interactions with the OpenWeatherMap API for retrieving historical weather data.
    """
    def __init__(self):
        """
        Initializes the API client with default parameters, including API key, location coordinates,
        retry count, and API endpoint.
        """
        self.api_key = os.environ['API_KEY']
        self.lat = 55.773239
        self.lon = 37.580463
        self.retries = 5
        self.timeout = 60  # seconds
        self.url_prefix = "https://api.openweathermap.org/data/3.0//onecall/timemachine?"

    def _get_query_path(self, dttm):
        """
        Constructs the query URL for fetching weather data at a specific time.

        Args:
            dttm (datetime): The datetime object for which weather data is requested.

        Returns:
            str: The full query URL.
        """
        ts = int(dttm.timestamp())
        logger.debug(
            "%slat=%d&lon=%d&dt=%d&appid=******&units=metric",
            self.url_prefix, self.lat, self.lon, ts)
        return f"{self.url_prefix}lat={self.lat}&lon={self.lon}&dt={ts}&appid={self.api_key}&units=metric"

    def get_raw_data(self, dttm):
        """
        Fetches raw weather data from the API for a specific datetime.

        Args:
            dttm (datetime): The datetime object for which weather data is requested.

        Returns:
            dict or None: The raw API response as a dictionary if successful; otherwise, None.
        """
        response = requests.get(url=self._get_query_path(dttm), timeout=self.timeout)
        remaining_retries = self.retries
        while response.status_code != 200 and remaining_retries > 0:
            remaining_retries -= 1
            response = requests.get(url=self._get_query_path(dttm), timeout=self.timeout)
        if response.status_code != 200:
            return None
        return response.json()

    def get_data_dict(self, raw_data):
        """
        Transforms raw weather data into a structured dictionary format.

        Args:
            raw_data (dict): The raw API response.

        Returns:
            dict or None: A processed dictionary containing relevant weather data, or None if input is invalid.
        """
        if raw_data is None:
            return None
        data = deepcopy(raw_data)
        if 'data' in data:
            for key, value in data['data'][0].items():
                data[key] = value
        if 'clouds' in data:
            data['clouds_all'] = data['clouds']
        if 'rain' in data:
            for key, value in data['rain'].items():
                data[f'rain_{key}'] = value
        if 'snow' in data:
            for key, value in data['snow'].items():
                data[f'snow_{key}'] = value
        if 'weather' in data and len(data['weather']) > 0:
            for key, value in data['weather'][0].items():
                data[f'weather_{key}'] = value
        return data


class Realtime:
    """
    Periodically retrieves weather data and uploads it to a database.
    """
    def __init__(self, table_nm=None):
        """
        Initializes the Realtime data collection process with default or user-specified configurations.

        Args:
            table_nm (str, optional): The name of the database table. Defaults to 'test_realtime_9'.
        """
        self.config = {
            'sleep_period_s': 3600,
            'table_nm': 'test_realtime_9'
        }
        if table_nm is not None:
            self.config['table_nm'] = table_nm
        self.api = Api()
        self.database = DatabaseManager(self.config['table_nm'])
        self.is_finished = False

    def upload_to_database(self, dict_data):
        """
        Uploads processed weather data to the database.

        Args:
            dict_data (dict): The dictionary of weather data to upload.
        """
        self.database.insert_data(dict_data)

    def run(self):
        """
        Continuously fetches and uploads weather data at periodic intervals until stopped.
        """
        while not self.is_finished:
            try:
                current_dttm = datetime.fromtimestamp(time.time()).replace(minute=0, second=0, microsecond=0)
                raw_data = self.api.get_raw_data(current_dttm)
                dict_data = self.api.get_data_dict(raw_data)
                current_dttm = current_dttm.strftime("%Y%m%d %H-%M-%S")
                if raw_data is not None:
                    self.upload_to_database(dict_data)
                    logger.debug("got data at %s", current_dttm)
                else:
                    logger.warning("failed data at %s", current_dttm)
            except RequestException:
                logger.error('exception while getting realtime data %s', traceback.format_exc())
            time.sleep(self.config['sleep_period_s'])


class TaskManager:
    """
    Manages the execution of the Realtime data collection process in a separate thread.
    """
    def __init__(self, table_nm):
        """
        Initializes the TaskManager with a specified database table name.

        Args:
            table_nm (str): The name of the database table for data storage.
        """
        self.realtime = Realtime(table_nm)
        self.thread = Thread(target=self.realtime.run)

    def start(self):
        """
        Starts the Realtime data collection process in a separate thread.
        """
        self.thread.start()

    def finish(self, timeout=None):
        """
        Stops the Realtime data collection process and waits for the thread to finish.

        Args:
            timeout (float, optional): The maximum time to wait for the thread to finish. Defaults to None.
        """
        self.realtime.is_finished = True
        self.thread.join(timeout)


if __name__ == "__main__":
    realtime = Realtime()
    realtime.run()
