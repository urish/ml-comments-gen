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
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(lstm_layer_size,), name='input_3')
    decoder_state_input_c = Input(shape=(lstm_layer_size,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return (encoder_model, decoder_model)

def decode_sequence(input_seq, encoder_model, decoder_model, comment_vocab_size, comment_tokenizer, max_comment_len):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, comment_vocab_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, comment_tokenizer.word_index['<start>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_comment = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        next_word = comment_tokenizer.index_word[sampled_token_index]
        if next_word == '<eos>':
            next_word = '\n'
        yield(next_word)

        # Exit condition: either hit max length
        # or find stop character.
        if (next_word == '<end>' or
           len(decoded_comment) > max_comment_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, comment_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

def predict_comment(ast_in, ast_tokenizer, comment_tokenizer, max_ast_len, max_comment_len, model, ast_vocab_size, comment_vocab_size, lstm_layer_size):
    encoder_model, decoder_model = get_encoder_decoder(model, lstm_layer_size)
    input_asts = ast_tokenizer.texts_to_sequences([ast_in])
    encoder_input_data = np.zeros(
        (len(input_asts), max_ast_len, ast_vocab_size),
        dtype='float32')
    for i, input_text in enumerate(input_asts):
        for t, token in enumerate(input_text):
            encoder_input_data[i, t, token] = 1.
    result = decode_sequence(encoder_input_data, encoder_model, decoder_model, comment_vocab_size, comment_tokenizer, max_comment_len)
    return ' '.join(list(result))

class ConditionalScope:
    def __init__(self, scope_factory, enabled = True):
        self.scope = scope_factory() if enabled else None
    
    def __enter__(self):
        if self.scope:
            self.scope.__enter__()

    def __exit__(self, type, value, traceback):
        if self.scope:
            self.scope.__exit__(type, value, traceback)
