from datetime import datetime
import psycopg2
import pytz
import pandas as pd
import numpy as np
from log import Logger

logger = Logger(__name__).get_logger()


class DatabaseManager:
    """
    A class to manage database operations.

    Attributes:
        table_nm (str): Name of the table for database operations.
        db_password (str): Password for the database connection.
        host (str): Host(s) for the database connection.
        port (int): Port for the database connection.
        db_nm (str): Database name.
        db_user (str): Username for the database connection.
        features (dict): Dictionary with features and their types.
    """

    def __init__(self, table_nm):
        """
        Initializes the DatabaseManager with a table name and default configurations.

        Args:
            table_nm (str): The default name of the table to manage.
        """
        self.table_nm = table_nm
        self.db_password = os.environ['PG_PASSWORD']
        self.host = 'rc1a-oe2h7ehfs86xa1fq.mdb.yandexcloud.net,rc1b-5k2553slvm58d9id.mdb.yandexcloud.net'
        self.port = 6432
        self.db_nm = 'Weather'
        self.db_user = 'db_user'
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
        self.create_table()

    def _execute(self, query, header=None, need_fetch=True):
        """
        Executes a given SQL query on the database.

        Args:
            query (str): The SQL query to execute.
            header (list, optional): Column headers for the result. Defaults to None.
            need_fetch (bool, optional): Whether to fetch results after execution. Defaults to True.

        Returns:
            pandas.DataFrame: DataFrame containing query results if need_fetch is True.
                Dataframe has columns from header if not None.
        """
        logger.debug(query)
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
                if need_fetch:
                    data = curs.fetchall()
                    df = pd.concat(
                        [pd.DataFrame(data, columns=header)], ignore_index=True)
                    return df
        return None

    def create_table(self, table_nm=None):
        """
        Creates a table in the database if it doesn't already exist.

        Args:
            table_nm (str, optional): The name of the table to create. Defaults to the instance's table_nm.
        """
        if table_nm is None:
            table_nm = self.table_nm
        self._execute(f"""CREATE TABLE IF NOT EXISTS {table_nm} (
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
                    )""", need_fetch=False)

    def drop_table(self, table_nm=None):
        """
        Drops a table from the database if it exists.

        Args:
            table_nm (str, optional): The name of the table to drop. Defaults to the instance's table_nm.
        """
        if table_nm is None:
            table_nm = self.table_nm
        self._execute(f"DROP TABLE IF EXISTS {table_nm}", need_fetch=False)

    def select(self, table_nm=None, header=None, additional_options=""):
        """
        Selects data from a table in the database.

        Args:
            table_nm (str, optional): The name of the table to select from. Defaults to the instance's table_nm.
            header (list, optional): Columns to select. Defaults to all columns defined in features.
            additional_options (str, optional): Additional SQL clauses for the SELECT query.
                Defaults to an empty string.

        Returns:
            pandas.DataFrame: DataFrame containing the selected data.
        """
        if header is None:
            header = [
                'city_name',
                'lat',
                'lon',
                'temp',
                'temp_min',
                'temp_max',
                'feels_like',
                'pressure',
                'humidity',
                'dew_point',
                'wind_speed',
                'wind_deg',
                'wind_gust',
                'clouds_all',
                'rain_1h',
                'rain_3h',
                'snow_1h',
                'snow_3h',
                'weather_id',
                'weather_main',
                'weather_description',
                'weather_icon',
                'visibility',
                'dt',
                'timezone',
                'sea_level',
                'grnd_level'
            ]

        if table_nm is None:
            table_nm = self.table_nm
        return self._execute(
            f'SELECT {",".join(header)} FROM public.{table_nm} {additional_options}',
            header)

    def _get_feature(self, data, feature_nm, feature_type):
        """
        Retrieves and formats a feature value from a data dictionary.

        Args:
            data (dict): The data dictionary containing feature values.
            feature_nm (str): The name of the feature to retrieve.
            feature_type (str): The type of the feature (e.g., 'str', 'num', 'timestamp').

        Returns:
            str: Formatted feature value for SQL insertion, or None if not applicable.
        """
        if feature_nm not in data:
            return None
        feature_value = data[feature_nm]
        formatted_value = None
        if isinstance(feature_value, (int, float)) and np.isnan(feature_value):
            formatted_value = 'NULL'
        elif feature_type == 'str':
            formatted_value = f"'{feature_value}'"
        elif feature_type == 'num':
            formatted_value = str(feature_value)
        elif feature_type == 'timestamp':
            if 'timezone' in data:
                dt = datetime.fromtimestamp(
                    int(feature_value),
                    tz=pytz.timezone(data['timezone'])
                )
                formatted_value = f"'{dt.isoformat(sep=' ')}'"
            else:
                dt = datetime.fromtimestamp(int(feature_value))
                formatted_value = f"'{dt.isoformat(sep=' ')}'"
        return formatted_value

    def insert_data(
            self,
            data_dict,
            feature_replacement_mapping=None,
            table_nm=None):
        """
        Inserts a single record into the database.

        Args:
            data_dict (dict): Data to insert.
            feature_replacement_mapping (dict, optional): Mapping of features in data_dict to database columns.
                Defaults to None.
            table_nm (str, optional): The name of the table to insert into.
                Defaults to the instance's table_nm.
        """
        if table_nm is None:
            table_nm = self.table_nm
        self.create_table(table_nm)
        headers = []
        values = []
        for k, v in self.features.items():
            if feature_replacement_mapping is not None:
                formatted_value = self._get_feature(
                    data_dict, feature_replacement_mapping.get(k, k), v)
            else:
                formatted_value = self._get_feature(data_dict, k, v)
            if formatted_value is not None:
                headers += [k]
                values += [formatted_value]
        self._execute(f'''INSERT INTO {table_nm} (
        {', '.join(headers)}) VALUES ({', '.join(values)})''', need_fetch=False)

    def insert_data_batch(
            self,
            data_list,
            feature_replacement_mapping=None,
            table_nm=None):
        """
        Inserts multiple records into the database.

        Args:
            data_list (list): List of data dictionaries to insert.
            feature_replacement_mapping (dict, optional): Mapping of features in data dictionaries to database columns.
                Defaults to None.
            table_nm (str, optional): The name of the table to insert into. Defaults to the instance's table_nm.
        """
        if table_nm is None:
            table_nm = self.table_nm
        logger.debug('inserting %d records into %s', len(data_list), table_nm)
        self.create_table(table_nm)
        q_values = []
        for data_dict in data_list:
            headers = []
            values = []
            for k, v in self.features.items():
                if feature_replacement_mapping is not None:
                    formatted_value = self._get_feature(
                        data_dict, feature_replacement_mapping.get(k, k), v)
                else:
                    formatted_value = self._get_feature(data_dict, k, v)
                if formatted_value is not None:
                    headers += [k]
                    values += [formatted_value]
            q_values += [f"({', '.join(values)})"]
        self._execute(f'''INSERT INTO {table_nm} (
        {', '.join(headers)}) VALUES {', '.join(q_values)}''', need_fetch=False)
