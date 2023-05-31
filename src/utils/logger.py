import sys
from datetime import timedelta


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        return 

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=seconds)).split(".")[0]
