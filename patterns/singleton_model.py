class ModelSingleton:

    _instance = None

    def __new__(cls, model):

        if cls._instance is None:
            cls._instance = model

        return cls._instance