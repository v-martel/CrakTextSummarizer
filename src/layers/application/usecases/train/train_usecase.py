from src.layers.application.services.ingestion.text_cleaner import add_start_end_token, clean_Digestible
from src.layers.application.services.ingestion.text_cleaner.helper_functions import get_recommended_length
from src.layers.application.services.ingestion.text_cleaner.tokenizer import SummarizerTokenizer
from src.layers.domain.model.digestible import Digestible
from src.layers.domain.model.headline_generator_lstm.summarizer_model import SummarizerModel
from src.layers.infrastructure.providers.chapi_provider import ChapiProvider
import warnings

warnings.filterwarnings("ignore")


class TrainUsecase:
    def __init__(self):
        self.chapi_provider = ChapiProvider()
        self.tokenizer_service = SummarizerTokenizer()

    def do(self):
        data = self._ingest()
        tokenized_data = self._digest(data)
        self._process(tokenized_data)

    def _ingest(self) -> Digestible:
        data = self.chapi_provider.execute_query()['PremiumVideos']['Data']

        inputs = list(map(lambda x: ' '.join(x['tags']), data))
        outputs = add_start_end_token(list(map(lambda x: x['title'], data)))

        return clean_Digestible(
            Digestible(
                inputs=inputs,
                outputs=outputs
            )
        )

    def _digest(self, data: Digestible):
        max_output_length = get_recommended_length(data.outputs)
        max_input_length = get_recommended_length(data.inputs)

        return self.tokenizer_service.tokenize_data(data, max_output_length, max_input_length)

    def _process(self, tokenized_data: dict):
        max_output_len, outputs_training, outputs_validation, outputs_voc_size, outputs_index_word, outputs_word_index = \
        tokenized_data['outputs']
        # print(outputs_word_index)

        max_input_len, inputs_training, inputs_validation, inputs_voc_size, inputs_index_word, inputs_word_index = \
        tokenized_data['inputs']
        print(max_input_len)

        print('outputs voc size = ', outputs_voc_size)
        print('input voc size = ', inputs_voc_size)

        print('sostok error? ', outputs_word_index['sostok'])

        summarizer_model = SummarizerModel(
            max_input_len=max_input_len,
            max_headline_len=max_output_len,

            articles_voc_size=inputs_voc_size,
            headlines_voc_size=outputs_voc_size,

            articles_training=inputs_training,
            headlines_training=outputs_training,

            articles_validation=inputs_validation,
            headlines_validation=outputs_validation,

            article_index_word=inputs_index_word,
            headline_index_word=outputs_index_word,

            article_word_index=inputs_word_index,
            headline_word_index=outputs_word_index
        )

        self.print_training_summaries(inputs_training, outputs_training, summarizer_model)

    def print_training_summaries(self, article_training, headline_training, model):
        for i in range(50):
            print("output training is : ", headline_training[i])
            self.print_training_summary(
                model.sequence_to_text(article_training[i]),
                model.decode_sequence(article_training[i].reshape(1, model.max_article_len)),
                model.sequence_to_summary(headline_training[i])
            )

    def print_training_summary(self, article, predicted_headline, headline):
        print("Input: ", article)
        print("Original output: ", headline)
        print("Predicted output: ", predicted_headline, "\n")
