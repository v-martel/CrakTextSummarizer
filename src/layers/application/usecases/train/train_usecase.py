from src.layers.application.services.ingestion.text_cleaner import add_start_end_token, clean_Digestible
from src.layers.application.services.ingestion.text_cleaner.helper_functions import get_recommended_length
from src.layers.application.services.ingestion.text_cleaner.tokenizer import SummarizerTokenizer
from src.layers.application.services.temp.data_getter.video_service import VideoService
from src.layers.domain.model.digestible import Digestible
from src.layers.domain.model.text_generator_lstm.summarizer_model import SummarizerModel
import warnings

warnings.filterwarnings("ignore")


class TrainUsecase:
    def __init__(self):
        self.chapi_provider = VideoService()
        self.tokenizer_service = SummarizerTokenizer()

    def do(self):
        data = self._ingest()
        tokenized_data = self._digest(data)
        self._process(tokenized_data)

    def _ingest(self) -> Digestible:
        data = self.chapi_provider.get_all_vids()

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

        max_input_len, inputs_training, inputs_validation, inputs_voc_size, inputs_index_word, inputs_word_index = \
        tokenized_data['inputs']

        summarizer_model = SummarizerModel(
            max_input_len=max_input_len,
            max_output_len=max_output_len,

            inputs_voc_size=inputs_voc_size,
            outputs_voc_size=outputs_voc_size,

            inputs_training=inputs_training,
            outputs_training=outputs_training,

            inputs_validation=inputs_validation,
            outputs_validation=outputs_validation,

            input_index_word=inputs_index_word,
            output_index_word=outputs_index_word,

            input_word_index=inputs_word_index,
            output_word_index=outputs_word_index
        )

        self.save_training_summaries(inputs_training, outputs_training, summarizer_model)

    def print_training_summaries(self, article_training, headline_training, model):
        for i in range(50):
            print("output training is : ", headline_training[i])
            self.print_training_summary(
                model.sequence_to_text(article_training[i]),
                model.decode_sequence(article_training[i].reshape(1, model.max_input_len)),
                model.sequence_to_summary(headline_training[i])
            )

    def save_training_summaries(self, article_training, headline_training, model):
        summary = ''
        for i in range(200):
            summary += self.get_training_summary(
                model.sequence_to_text(article_training[i]),
                model.sequence_to_summary(headline_training[i]),
                model.decode_sequence(article_training[i].reshape(1, model.max_input_len))
            )

        file = open('./data/results/results.txt', 'w')
        file.write(summary)
        file.close()


    def print_training_summary(self, article, predicted_headline, headline):
        print("Input: ", article)
        print("Original output: ", headline)
        print("Predicted output: ", predicted_headline, "\n")

    def get_training_summary(self, article, headline, predicted_headline):
        return """
            Input: %s
            Original output: %s
            Predicted output: %s \n
        """ % (article, headline, predicted_headline)
