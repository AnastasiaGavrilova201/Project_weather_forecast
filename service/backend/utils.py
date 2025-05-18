import pandas as pd

from log import Logger

logger = Logger(__name__).get_logger()


class CsvToDatabase:
    """
    A class for handling the process of transforming and uploading CSV data into a database.

    Attributes:
        db_manager (DatabaseManager): A database manager instance for interacting with the database.
    """
    def __init__(self, db_manager):
        """
        Initializes the CsvToDatabase class with a database manager.

        Args:
            db_manager (DatabaseManager): An instance of the DatabaseManager class to handle database operations.
        """
        self.db_manager = db_manager

    def timezones_to_str(self, df, column_name, inplace=False):
        """
        Converts timezone offsets in a DataFrame column to string representations of timezones.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to convert.
            column_name (str): The name of the column with timezone offsets.
            inplace (bool, optional): Whether to modify the DataFrame in place. Defaults to False.

        Returns:
            pd.DataFrame: The updated DataFrame with timezone strings.
        """
        offset_to_tz = {
            10800: "Europe/Moscow",
            0: "UTC",
        }
        if inplace:
            df_new = df
        else:
            df_new = df.copy()
        df_new[column_name] = df_new[column_name].map(offset_to_tz).fillna("Europe/Moscow")
        return df_new

    def upload_csv_to_db(self, filename, table_nm, batch_size=256):
        """
        Uploads data from a CSV file into a database table in batches.

        Args:
            filename (str): The path to the CSV file to upload.
            table_nm (str): The name of the database table to insert data into.
            batch_size (int, optional): The number of records to include in each batch. Defaults to 256.
        """
        logger.debug('uploading data from %s to table %s', filename, table_nm)
        df = pd.read_csv(filename)
        if df['timezone'].dtype != str:
            self.timezones_to_str(df, 'timezone', inplace=True)
        dict_records = df.to_dict(orient='records')
        records = []
        for record in dict_records:
            if len(records) >= batch_size:
                self.db_manager.insert_data_batch(records, table_nm=table_nm)
                records = []
            else:
                records += [record]
        if len(records) >= 0:
            self.db_manager.insert_data_batch(records, table_nm=table_nm)
