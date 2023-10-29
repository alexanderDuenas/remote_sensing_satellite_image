# TODO: Adicionar novas configurações neste arquivo. Evitar subir para o Git links ou credenciais de acesso.
#       Nesses casos, utilizar o arquivo .env, que nunca deve ser enviado para o Git.

import os

from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

api_config = dict(
    SERVER_HOST=os.environ.get("API_SERVER_HOST"),
    SERVER_PORT=5000
)

processing_thread_config = dict(
    NAME=os.environ.get("THREAD_NAME")
)

jhipster_registry_config = dict(
    SERVER_HOST=os.environ.get("JHIPSTER_REGISTRY_SERVER_HOST"),
    USER=os.environ.get("JHIPSTER_REGISTRY_USER"),
    PASSWORD=os.environ.get("JHIPSTER_REGISTRY_PASSWORD"),
    ENABLE=False
)

minio_config = dict(
    SERVER_HOST=os.environ.get("MINIO_SERVER_HOST"),
    ACCESS_KEY=os.environ.get("MINIO_ACCESS_KEY"),
    SECRET_KEY=os.environ.get("MINIO_SECRET_KEY"),
    SECURE=os.environ.get("MINIO_SECURE"),
    URL=os.environ.get("MINIO_URL"),
    ENABLE=False
)
