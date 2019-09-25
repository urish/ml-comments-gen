import pickle
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from parameters import UNITS


def max_length(seqs, max_seq_len):
    length = max(len(s) for s in seqs)
    return length if length < max_seq_len else max_seq_len


def save_tokenizer(tokenizer, file_path):
    with open(file_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer successfully saved ({})".format(file_path))


def load_tokenizer(file_path):
    with open(file_path, "rb") as handle:
        tokenizer = pickle.load(handle)
        print("Tokenizer loaded ({})".format(file_path))
        return tokenizer


def get_encoder_decoder(model, lstm_layer_size):
    encoder_inputs = model.input[0]  # input_1
    _, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(lstm_layer_size,), name="input_3")
    decoder_state_input_c = Input(shape=(lstm_layer_size,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embeddings = model.layers[3](decoder_inputs)
    decoder_lstm = model.layers[5]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embeddings, initial_state=decoder_states_inputs
    )

    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return (encoder_model, decoder_model)


def decode_sequence(
    input_seq,
    encoder_model,
    decoder_model,
    comment_vocab_size,
    comment_tokenizer,
    max_comment_len,
    comment_start_token,
    comment_end_token,
):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = comment_start_token

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    prev_word = ''
    for i in range(max_comment_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == comment_end_token:
            break
        yield comment_tokenizer.decode([sampled_token_index])

        # Update the target sequence (of length 1).
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]


def predict_comment(
    ast_in,
    ast_tokenizer,
    comment_tokenizer,
    max_ast_len,
    max_comment_len,
    model,
    ast_vocab_size,
    comment_vocab_size,
    lstm_layer_size,
    comment_start_token,
    comment_end_token,
    ast_start_token,
    ast_end_token,
):
    encoder_model, decoder_model = get_encoder_decoder(model, lstm_layer_size)
    input_asts = [[ast_start_token] + ast_tokenizer.encode(ast_in) + [ast_end_token]]

    encoder_input_data = np.zeros(
        (len(input_asts), max_ast_len), dtype="float32"
    )

    for i, input_text in enumerate(input_asts):
        for t, token in enumerate(input_text[:max_ast_len]):
            encoder_input_data[i, t] = token

    result = decode_sequence(
        encoder_input_data,
        encoder_model,
        decoder_model,
        comment_vocab_size,
        comment_tokenizer,
        max_comment_len,
        comment_start_token,
        comment_end_token,
    )
    return "".join(list(result))


class ConditionalScope:
    def __init__(self, scope_factory, enabled=True):
        self.scope = scope_factory() if enabled else None

    def __enter__(self):
        if self.scope:
            self.scope.__enter__()

    def __exit__(self, type, value, traceback):
        if self.scope:
            self.scope.__exit__(type, value, traceback)
