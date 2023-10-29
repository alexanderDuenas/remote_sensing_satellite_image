from app.src.resources import root_resource, example_resource


def init_resources(app):
    app.include_router(root_resource.root_api)
    app.include_router(example_resource.example_api)
    # TODO: Registrar novos recursos (arquivos de endpoints) aqui.
