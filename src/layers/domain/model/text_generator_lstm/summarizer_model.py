import numpy as np
# import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from src.layers.domain.model.text_generator_lstm.attention_layer import AttentionLayer


class SummarizerModel:
    def __init__(self,

                 max_input_len,
                 max_output_len,

                 inputs_voc_size,
                 outputs_voc_size,

                 inputs_training,
                 outputs_training,

                 inputs_validation,
                 outputs_validation,

                 input_index_word,
                 output_index_word,

                 input_word_index,
                 output_word_index,
                 ):

        backend.clear_session()

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        latent_dim = 300
        embedding_dim = 120

        # Encoder
        encoder_inputs = Input(shape=(self.max_input_len,))

        # embedding layer
        enc_emb = Embedding(inputs_voc_size, embedding_dim, trainable=True)(encoder_inputs)

        # encoder LSTM 1
        encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        # encoderLSTM 2
        encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # encoder LSTM 3
        encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        # Set up the decoder, using encoder_states as initial state
        decoder_inputs = Input(shape=(None,))

        # embedding layer
        dec_emb_layer = Embedding(outputs_voc_size, embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
            dec_emb, initial_state=[state_h, state_c])

        # Attention Layer
        attn_layer = AttentionLayer(name="attention_layer")
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention output and decode LSTM output
        decoder_concat_input = Concatenate(axis=-1, name="concat_layer")([decoder_outputs, attn_out])

        # Dense Layer
        decoder_dense = TimeDistributed(Dense(outputs_voc_size, activation="softmax"))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.summary()

        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        self.es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)

        history = self._train(
            inputs_training,
            outputs_training,
            inputs_validation,
            outputs_validation
        )

        self.reverse_target_word_index = output_index_word
        self.reverse_source_word_index = input_index_word
        self.target_word_index = output_word_index

        # Encode the input sequence to get the feature vector
        self.encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_input_len, latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[
            decoder_state_input_h,
            decoder_state_input_c
        ])

        # attention inference
        attn_out_inf, attn_states_inf = attn_layer([
            decoder_hidden_state_input,
            decoder_outputs2
        ])
        decoder_inf_concat = Concatenate(axis=-1, name="concat")([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_inf_concat)

        # Final decoder model
        self.decoder_model = Model(
            [decoder_inputs] + [
                decoder_hidden_state_input,
                decoder_state_input_h,
                decoder_state_input_c
            ],
            [decoder_outputs2] + [state_h2, state_c2]
        )

    def _train(self, inputs_training,
               outputs_training,
               inputs_validation,
               outputs_validation):
        return self.model.fit(
            [inputs_training, outputs_training[:, :-1]],
            outputs_training.reshape(outputs_training.shape[0], outputs_training.shape[1], 1)[:, 1:],
            epochs=10, callbacks=[self.es], batch_size=75,
            validation_data=(
                [inputs_validation, outputs_validation[:, :-1]],
                outputs_validation.reshape(outputs_validation.shape[0], outputs_validation.shape[1], 1)[:, 1:]
            )
        )

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = self.target_word_index['sostok']

        decoded_sentence = ''
        for i in range(self.max_output_len - 1):
            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            prediction_weights = output_tokens[0, -1, :]
            prediction_weights[2] = prediction_weights[2] * 0.2 * i
            prediction_weights[0] = 0

            sampled_token_index = np.argmax(prediction_weights)

            sampled_token = self.reverse_target_word_index[sampled_token_index]
            print(sampled_token)

            if sampled_token != 'eostok':
                decoded_sentence += ' ' + sampled_token
            else:
                break

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            e_h, e_c = h, c

        return decoded_sentence[1:]

    def sequence_to_summary(self, input_sequence):
        return " ".join([
            self.reverse_target_word_index[i] for i in input_sequence if
            i != 0 and
            i != self.target_word_index["sostok"] and
            i != self.target_word_index["eostok"]
        ])

    def sequence_to_text(self, input_sequence):
        return " ".join([self.reverse_source_word_index[i] for i in input_sequence if i != 0])
