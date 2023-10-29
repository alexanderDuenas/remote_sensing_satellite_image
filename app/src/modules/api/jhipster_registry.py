from jhipster import JHipsterRegistry

from config import jhipster_registry_config


def init_registry():
    if jhipster_registry_config.get('ENABLE'):
        return JHipsterRegistry(
            url=jhipster_registry_config.get("SERVER_HOST"),
            credentials=(jhipster_registry_config.get("USER"), jhipster_registry_config.get("PASSWORD")),
            application="application",
            profile="prod"
        )
