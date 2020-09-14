from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.layers.domain.model.digestible import Digestible


class SummarizerTokenizer:
    def tokenize_data(self, digestible: Digestible, max_output_len: int, max_input_len: int) -> dict:
        outputs_training, outputs_validation, inputs_training, inputs_validation = self.split_training_validation(
            digestible
        )

        inputs_tokenizer = Tokenizer()
        outputs_tokenizer = Tokenizer()

        inputs_tokenizer.fit_on_texts(inputs_training)
        outputs_tokenizer.fit_on_texts(outputs_training)

        input_rare_count, input_total_count, input_rare_frequency, input_total_frequency = self.handle_rare_words(
            inputs_tokenizer, 0)
        output_rare_count, output_total_count, output_rare_frequency, output_total_frequency = self.handle_rare_words(
            outputs_tokenizer, 0)

        # prepare a tokenizer for reviews on training data
        inputs_tokenizer = Tokenizer(num_words=input_total_count - input_rare_count)
        inputs_tokenizer.fit_on_texts(inputs_training)

        outputs_tokenizer = Tokenizer(num_words=output_total_count - output_rare_count)
        outputs_tokenizer.fit_on_texts(outputs_training)
        print('why is sostok not in there?', inputs_validation)

        # convert text sequences into integer sequences and padding zero upto maximum length
        inputs_training = self.tokenize_input(inputs_tokenizer, max_input_len, inputs_training)
        inputs_validation = self.tokenize_input(inputs_tokenizer, max_input_len, inputs_validation)

        outputs_training = self.tokenize_output(outputs_tokenizer, max_output_len, outputs_training)
        outputs_validations = self.tokenize_output(outputs_tokenizer, max_output_len, outputs_validation)

        # size of vocabulary
        articles_voc_size = inputs_tokenizer.num_words + 1
        headlines_voc_size = outputs_tokenizer.num_words + 1

        return {
            "outputs": (
                max_output_len,
                outputs_training,
                outputs_validations,
                headlines_voc_size,
                outputs_tokenizer.index_word,
                outputs_tokenizer.word_index
            ),
            "inputs": (
                max_input_len,
                inputs_training,
                inputs_validation,
                articles_voc_size,
                inputs_tokenizer.index_word,
                inputs_tokenizer.word_index
            )
        }

    def tokenize_output(self, output_tokenizer: Tokenizer, max_output_len: int, output_list: [str]):
        new_output_list = output_tokenizer.texts_to_sequences(output_list)
        return pad_sequences(new_output_list, maxlen=max_output_len, padding="post")

    def tokenize_input(self, input_tokenizer: Tokenizer, max_input_len: int, input_list: [str]):
        new_input_list = input_tokenizer.texts_to_sequences(input_list)
        return pad_sequences(new_input_list, maxlen=max_input_len, padding="post")

    def handle_rare_words(self, tokenizer: Tokenizer, threshold: int) -> (int, int, int, int):
        count = 0
        total_count = 0
        frequency = 0
        total_frequency = 0

        for key, value in tokenizer.word_counts.items():
            total_count += 1
            total_frequency += value
            if value < threshold:
                count += 1
                frequency += value

        return count, total_count, frequency, total_frequency

    def split_training_validation(self, digestible: Digestible) -> ([str], [str], [str], [str]):
        outputs_training, outputs_validation, inputs_training, inputs_validation = train_test_split(
            digestible.outputs,
            digestible.inputs,
            test_size=0.15,
            random_state=0,
            shuffle=True)

        return outputs_training, outputs_validation, inputs_training, inputs_validation
