from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Configuration:
    def __init__(self):
        self.layers = LayersConfiguration()
        self.presentation = None


class LayersConfiguration:
    def __init__(self):
        self.infrastructure = InfrastructureConfiguration()


class InfrastructureConfiguration:
    def __init__(self):
        self.providers = ProvidersConfiguration()


class ProvidersConfiguration:
    def __init__(self):
        self.chapi_provider = ChapiProvider()


class ChapiProvider:
    def __init__(self):
        self.chapi_url = getenv('CHAPI_URL')
        self.chapi_token = getenv('CHAPI_TOKEN')


# should be used as a singleton
configuration = Configuration()
