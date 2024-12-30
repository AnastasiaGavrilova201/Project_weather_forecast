import logging
from logging.handlers import RotatingFileHandler
import os

class Logger:
    def __init__(self, name: str = __name__, log_file: str = "app.log", log_directory: str = "./log", max_bytes: int = 1024**2, file_count: int = 5):
        """
        Initializes a logger with rotation.

        Args:
            name (str): Logger name.
            log_file (str): File to save logs.
            log_directory (str): Directory to save logs. 
            max_bytes (int): Max size of a log file before rotating. Defaults to 1 MB.
            backup_count (int): Number of backup files to keep. Defaults to 5.
        """
        os.makedirs(log_directory, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = RotatingFileHandler(
                f'{log_directory}/{log_file}', maxBytes=max_bytes, backupCount=file_count
            )
            formatter = logging.Formatter(
                "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        """Returns the configured logger."""
        return self.logger
