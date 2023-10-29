import threading

from app.src.services.example_service import start_processing_service
from config import processing_thread_config


class AnalysisThread(threading.Thread):

    def __init__(self, params):
        threading.Thread.__init__(self)
        self.name = processing_thread_config.get("NAME")
        self.params = params

    def run(self):
        try:
            start_processing_service(self.params)
        except Exception as exception:
            print(f'Something went wrong: {exception}')
        finally:
            print(f'Thread {self.name} stopped!')
