import csv
import os
import requests
import shutil
from datetime import datetime
import time
import traceback
import psycopg2
import pytz
from copy import deepcopy

class Config:
    def __init__(self):
        # general
        self.sleep_period_s = 900
        self.lat = 55.773239
        self.lon = 37.580463
        self.retries = 5
        self.api_key = os.environ['API_KEY']
        self.features = {
            'dt': 'timestamp',
            'timezone': 'str',
            'city_name': 'str',
            'lat': 'num',
            'lon': 'num',
            'temp': 'num',
            'visibility': 'num',
            'dew_point': 'num',
            'feels_like': 'num',
            'temp_min': 'num',
            'temp_max': 'num',
            'pressure': 'num',
            'sea_level': 'num',
            'grnd_level': 'num',
            'humidity': 'num',
            'wind_speed': 'num',
            'wind_deg': 'num',
            'wind_gust': 'num',
            'rain_1h': 'num',
            'rain_3h': 'num',
            'snow_1h': 'num',
            'snow_3h': 'num',
            'clouds_all': 'num',
            'weather_id': 'num',
            'weather_main': 'str',
            'weather_description': 'str',
            'weather_icon': 'str',
        }
        
        
        # db
        self.db_nm = 'Weather'
        self.table_nm = 'realtime'
        self.db_password = os.environ['PG_PASSWORD']
        self.db_port = 6432
        self.db_host = os.environ['PG_HOST']
        self.db_user = 'db_user'
        
        # file
        self.data_filename = 'Moscow_realtime.csv'
        self.store_to_file = False
        self.init_file = True
        self.header = [key for key in self.features.keys()] + ['dt_iso']

class Api:
    def __init__(self, config):
        self.config = config
        self.query_path = f"https://api.openweathermap.org/data/3.0/onecall?lat={self.config.lat}&lon={self.config.lon}&exclude=minutely,hourly,daily,alerts&appid={self.config.api_key}&units=metric"

    def get_raw_data(self):
        response = requests.get(url=self.query_path)
        remaining_retries = self.config.retries
        while response.status_code != 200 and remaining_retries > 0:
            remaining_retries -= 1
            response = requests.get(url=self.query_path)
        if response.status_code != 200:
            return None
        return response.json()
    
    def get_data_dict(self, raw_data):
        if raw_data is None:
            return None
        data = deepcopy(raw_data)
        if 'current' in data:
            for key, value in data['current'].items():
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
    
    def get_data_list(self, raw_data):
        if raw_data is None:
            return None
        dict_data = self.get_data_dict(raw_data)
        data = [dict_data.get(header) for header in self.config.header]
        return data

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.table_nm = config.table_nm
        self.db_password = config.db_password
        self.host = config.db_host
        self.port = config.db_port
        self.db_nm = config.db_nm
        self.db_user = config.db_user
        self.features = config.features
        self._create_table()
    
    def _execute(self, query):
        with psycopg2.connect(f"""
        host={self.host}
        port={self.port}
        sslmode=verify-full
        dbname={self.db_nm}
        user={self.db_user}
        password={self.db_password}
        target_session_attrs=read-write
    """) as conn:
            with conn.cursor() as curs:
                curs.execute(query)
    
    def _create_table(self):
        self._execute(f"""CREATE TABLE IF NOT EXISTS {self.table_nm} (
                    city_name           varchar(40),
                    lat                 decimal(8, 5),
                    lon                 decimal(8, 5),
                    temp                decimal(8, 4),
                    temp_min            decimal(8, 4),
                    temp_max            decimal(8, 4),
                    feels_like          decimal(8, 4),
                    pressure            decimal(8, 3),
                    humidity            decimal(8, 4),
                    dew_point           decimal(8, 4),
                    wind_speed          decimal(8, 4),
                    wind_deg            decimal(8, 4),
                    wind_gust           decimal(8, 4),
                    clouds_all          decimal(8, 4),
                    rain_1h             decimal(8, 2),
                    rain_3h             decimal(8, 2),
                    snow_1h             decimal(8, 2),
                    snow_3h             decimal(8, 2),
                    weather_id          integer,
                    weather_main        varchar(40),
                    weather_description varchar(40),
                    weather_icon        varchar(40),
                    visibility          decimal(10, 3),
                    dt                  timestamptz,
                    timezone            varchar(40),
                    sea_level           decimal(8, 3),
                    grnd_level          decimal(8, 3)
                    )""")
                
    def _get_feature(self, data, feature_nm):
        if feature_nm not in data or feature_nm not in self.features:
            return None
        feature_value = data[feature_nm]
        if self.features[feature_nm] == 'str':
            return f"'{feature_value}'"
        elif self.features[feature_nm] == 'num':
            return str(feature_value)
        elif self.features[feature_nm] == 'timestamp':
            if 'timezone' in data:
                dt = datetime.fromtimestamp(int(feature_value),
                                       tz=pytz.timezone(data['timezone'])
                                      )
                return f"'{dt.isoformat(sep=' ')}'"
            else:
                dt = datetime.fromtimestamp(int(feature_value))
                return f"'{dt.isoformat(sep=' ')}'"
        return None
    
    def insert_data(self, data_dict):
        headers = []
        values = []
        for k, v in self.features.items():
            formatted_value = self._get_feature(data_dict, k)
            if formatted_value is not None:
                headers += [k]
                values += [formatted_value]
        self._execute(f'''INSERT INTO {self.table_nm} (
        {', '.join(headers)}) VALUES ({', '.join(values)})''')

class TaskManager:
    def __init__(self, config):
        self.config = config
        self.api = Api(config)
        self.db = DatabaseManager(config)
        if self.config.init_file:
            self.init_file()
        self.counter = 0

    def init_file(self):
        if not self.config.store_to_file:
            return
        with open(self.config.data_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.config.header)

    def upload_to_file(self, data):
        if not self.config.store_to_file:
            return
        with open(self.config.data_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def upload_to_db(self, dict_data):
        for _ in range(self.config.retries):
            try:
                self.db.insert_data(dict_data)
                return
            except Exception:
                print('exception while updating db')
                traceback.print_exc()

    def run(self):
        while(True):
            try:
                raw_data = self.api.get_raw_data()
                dict_data = self.api.get_data_dict(raw_data)
                data = self.api.get_data_list(raw_data)
                current_dttm = datetime.fromtimestamp(time.time()).strftime("%Y%m%d %H-%M-%S")
                if data is not None:
                    self.upload_to_file(data)
                    self.upload_to_db(dict_data)
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