class CheckConfig:

    def __init__(self, model, registry, minio):
        self.model = model
        self.registry = registry
        self.minio = minio

    def get_config(self):
        return {
            "model": self.model,
            "registry": self.registry,
            "minio": self.minio
        }
