import logging
import sys
import os

class Logger():
    _logger_file_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
    _log_file_path = os.path.join(_logger_file_path, "output.log")
    level = logging.INFO
    output_file_handler = logging.FileHandler(_log_file_path)
    output_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.output_file_handler)
        self.logger.addHandler(self.stdout_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)