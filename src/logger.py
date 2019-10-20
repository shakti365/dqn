from datetime import datetime
import pytz

class Logger:

    def __init__(self):
        uid = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H:%M:%S")
        self.logger_file = open(f"logs/{uid}.log", "w")

    def log(self, text):
        time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H:%M:%S")
        self.logger_file.write(f"{time}: {text}\n")

    def close(self):
        self.log.close()
