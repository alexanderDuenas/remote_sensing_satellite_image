from app.src.analytics.processing_controller import ProcessingController


# TODO: Caso seja necessário, implementar lógica de tratamento dos dados antes da inicialização do algoritmo.
def start_processing_service(params):
    if params is not None:
        processing = ProcessingController(params)
        processing.run()
        processing.stop()
