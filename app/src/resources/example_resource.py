import logging

from fastapi import APIRouter, Body, HTTPException

from app.src.modules.analysis_thread.analysis_thread import AnalysisThread
from app.src.utils.thread_utils import get_thread_if_exist
from config import processing_thread_config
from docs.api.example_body_request import example_body_request

logger = logging.getLogger('template_vc')

example_api = APIRouter()


@example_api.get('/')
async def index():
    return {"message": "Example Resource"}


# TODO: Neste caso, como a thread possui sempre o mesmo nome, o código irá limitar em no máximo um processamento por
#       vez. Caso seja necessário executar vários processamentos ao mesmo tempo, o ideal seria alterar a lógica para
#       utilizar um nome dinâmico para a thread, como um ID único ou algo semelhante.
@example_api.post('/start')
async def start(request: dict = Body(..., example=example_body_request)):
    logger.info(f'Requisição para iniciar o exemplo: {request}')
    """
        Exemplo de documentação da API: Aqui você pode comentar como a API do projeto deve ser utilizada, quais 
        parâmetros devem ser passados, qual é o retorno esperado, etc.
    """
    running_thread = get_thread_if_exist(processing_thread_config.get("NAME"))
    if running_thread is not None:
        raise HTTPException(status_code=503, detail="Uma thread de análise já está em processamento.")
    else:
        new_thread = AnalysisThread(request.get("counter_limit"))
        new_thread.start()
        return {"message": "Uma nova thread de análise foi iniciada."}
