import os
import requests
from datetime import datetime
import time
import traceback
from copy import deepcopy

from postgres import DatabaseManager
from log import Logger

logger = Logger(__name__).get_logger()


class Api:
    def __init__(self):
        self.api_key = os.environ['API_KEY']
        self.lat = 55.773239
        self.lon = 37.580463
        self.retries = 5

    def _get_query_path(self, dttm):
        ts = int(dttm.timestamp())
        return f"https://api.openweathermap.org/data/3.0//onecall/timemachine?lat={self.lat}&lon={self.lon}&dt={ts}&appid={self.api_key}&units=metric"

    def get_raw_data(self, dttm):
        response = requests.get(url=self._get_query_path(dttm))
        remaining_retries = self.retries
        while response.status_code != 200 and remaining_retries > 0:
            remaining_retries -= 1
            response = requests.get(url=self._get_query_path(dttm))
        if response.status_code != 200:
            return None
        return response.json()

    def get_data_dict(self, raw_data):
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

class TaskManager:
    def __init__(self):
        self.config = {
            'sleep_period_s': 900,
            'table_nm': 'test_realtime_9',
            'retries': 5
        }
        self.api = Api()
        self.database = DatabaseManager(self.config['table_nm'])

    def upload_to_database(self, dict_data):
        for _ in range(self.config['retries']):
            try:
                self.database.insert_data(dict_data)
                return
            except Exception:
                logger.error('exception while updating database %s', traceback.format_exc())

    def run(self):
        while(True):
            try:
                current_dttm = datetime.fromtimestamp(time.time()).replace(minute=0, second=0, microsecond=0)
                raw_data = self.api.get_raw_data()
                dict_data = self.api.get_data_dict(raw_data)
                current_dttm = current_dttm.strftime("%Y%m%d %H-%M-%S")
                if raw_data is not None:
                    self.upload_to_database(dict_data)
                    print(f"got data at %s", current_dttm)
                else:
                    print(f"failed data at %s", current_dttm)
            except Exception:
                logger.error('exception while getting realtime data %s', traceback.format_exc())
            time.sleep(self.config['sleep_period_s'])

task_manager = TaskManager()
task_manager.run()