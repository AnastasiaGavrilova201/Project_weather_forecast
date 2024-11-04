import csv
import os
import requests
import shutil
from datetime import datetime
import time
import traceback

class Config:
    def __init__(self):
        self.data_filename = 'Moscow_realtime.csv'
        self.init_file = True
        self.sleep_period_s = 900
        self.api_key = os.environ['API_KEY']
        self.backup_every_iter = 10
        self.lat = 55.773239
        self.lon = 37.580463
        self.retries = 5
        self.file_header = ['dt',
                            'dt_iso',
                            'timezone',
                            'city_name',
                            'lat',
                            'lon',
                            'temp',
                            'visibility',
                            'dew_point',
                            'feels_like',
                            'temp_min',
                            'temp_max',
                            'pressure',
                            'sea_level',
                            'grnd_level',
                            'humidity',
                            'wind_speed',
                            'wind_deg',
                            'wind_gust',
                            'rain_1h',
                            'rain_3h',
                            'snow_1h',
                            'snow_3h',
                            'clouds_all',
                            'weather_id',
                            'weather_main',
                            'weather_description',
                            'weather_icon',
                            ]

class Api:
    def __init__(self, config):
        self.config = config
        self.query_path = f"https://api.openweathermap.org/data/3.0/onecall?lat={self.config.lat}&lon={self.config.lon}&exclude=minutely,hourly,daily,alerts&appid={self.config.api_key}&units=metric"

    def get_data_dict(self):
        response = requests.get(url=self.query_path)
        remaining_retries = self.config.retries
        while response.status_code != 200 and remaining_retries > 0:
            remaining_retries -= 1
            response = requests.get(url=self.query_path)
        if response.status_code != 200:
            return None
        return response.json()
    
    def get_data_list(self):
        raw_data = self.get_data_dict()
        if raw_data is None:
            return None
        data = []
        if 'current' in raw_data:
            for key, value in raw_data['current'].items():
                raw_data[key] = value
        if 'clouds' in raw_data:
            raw_data['clouds_all'] = raw_data['clouds']
        if 'rain' in raw_data:
            for key, value in raw_data['rain'].items():
                raw_data[f'rain_{key}'] = value
        if 'snow' in raw_data:
            for key, value in raw_data['snow'].items():
                raw_data[f'snow_{key}'] = value
        if 'weather' in raw_data and len(raw_data['weather']) > 0:
            for key, value in raw_data['weather'][0].items():
                raw_data[f'weather_{key}'] = value
        data = [raw_data.get(header) for header in self.config.file_header]
        return data
    

class TaskManager:
    def __init__(self, config):
        self.config = config
        self.api = Api(config)
        if self.config.init_file:
            self.init_file()
        self.counter = 0

    def init_file(self):
        with open(self.config.data_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.config.file_header)
    
    def backup(self):
        current_dttm = datetime.fromtimestamp(time.time()).strftime("%Y%m%d %H-%M-%S")
        backup_file = f"./backup/{current_dttm}_{self.config.data_filename}"
        print(f"backup to {backup_file}")
        shutil.copy(self.config.data_filename, backup_file)
    
    def run(self):
        while(True):
            try:
                if self.counter % self.config.backup_every_iter == 0:
                    self.backup()
                data = self.api.get_data_list()
                current_dttm = datetime.fromtimestamp(time.time()).strftime("%Y%m%d %H-%M-%S")
                if data is not None:
                    with open(self.config.data_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)
                    print(f"got data at {current_dttm}")
                else:
                    print(f"failed data at {current_dttm}")
            except Exception:
                traceback.print_exc()
            self.counter += 1
            time.sleep(self.config.sleep_period_s)

config = Config()
task_manager = TaskManager(config)
task_manager.run()