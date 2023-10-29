import datetime
import logging

from app.src.analytics.modules import counter_example
from app.src.analytics.modules.counter_example import counter_example
from app.src.ditcs.example_result import ExampleResult

logger = logging.getLogger('template_vc')


class ProcessingController:

    def __init__(self, params):
        self.params = params

    # TODO: Implementar/chamar lógica do algoritmo.
    def run(self):
        counter_example(self.params)

    # TODO: Caso seja necessário, implementar lógica de tratamento dos dados antes da finalizar o algoritmo.
    #       Neste caso, foi utilizado um dicionário para popular o objeto de retorno (ExampleResult).
    def stop(self):
        example_result = ExampleResult(
            datetime.datetime.now(),
            self.params,
            "First Value",
            "Second Value",
            "Message"
        )
        logger.info(f"### Resultado final: {example_result.get_result()}")
