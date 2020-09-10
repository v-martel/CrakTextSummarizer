# from src.layers.application.usecases.model_training_and_tests.tests import test_training
from src.layers.infrastructure.providers.chapi_provider import ChapiProvider

# def main(): return test_training()

if __name__ == "__main__":
    print('Hello world!')
    chapiProvider = ChapiProvider()
    print(chapiProvider.execute_query())
    # main()
