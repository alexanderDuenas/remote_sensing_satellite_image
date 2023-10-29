import requests

from config import processing_thread_config


def send_result(example_result):
    try:
        request = requests.post(
            f'{processing_thread_config.get("SERVER_HOST")}/etc',
            json=example_result
        )
        if request.ok:
            print(f'Resultado enviado: {request.text}')
    except Exception as exception:
        raise exception
