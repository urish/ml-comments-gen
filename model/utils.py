import pickle
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from parameters import UNITS


def max_length(seqs, max_seq_len):
    length = max(len(s) for s in seqs)
    return length if length < max_seq_len else max_seq_len


def prepare_dataset(name, raw_x1, raw_x2, max_seq_len, x2_vocab_size):
    max_len_x1 = max_length(raw_x1, max_seq_len)
    max_len_x2 = max_length(raw_x2, max_seq_len)

    print("Max X1", max_len_x1)
    print("Max X2", max_len_x2)

    max_len = max(max_len_x1, max_len_x2)

    x1, x2, y = list(), list(), list()
    for x2_idx, seq in enumerate(raw_x2):

        x1_encoded = pad_sequences(
            [raw_x1[x2_idx]], maxlen=max_len, padding="post", truncating="post"
        )[0]

        for i in range(1, len(seq)):
            # add function signature
            x1.append(x1_encoded)

            # add the entire sequence to the input and only keep the next word for the output
            in_seq, out_seq = seq[:i], seq[i]

            # apply padding and encode sequence
            in_seq = pad_sequences(
                [in_seq], maxlen=max_len, padding="post", truncating="post"
            )[0]

            # one hot encode output sequence
            out_seq = to_categorical([out_seq], num_classes=x2_vocab_size)[0]
            y.append(out_seq)

            # cut the input seq to fixed length
            x2.append(in_seq)

    x1, x2, y = np.array(x1), np.array(x2), np.array(y)
    return x1, x2, y, max_len


"""
This function can be used to debug the preprocessed dataset and print out
encoded inputs as well as the label.
Be aware of calling this function on a large dataset as it will produce dozens
of print statemenets.
"""


def check_encoding(x1, x2, y, ast_idx2word, comment_idx2word):
    print("--- Check Encoding ---")
    for idx, ast in enumerate(x1):
        x1_decoded = ""
        x2_decoded = ""
        y_decoded = ""

        # decode one hot encoding for x1
        for x1_encoded in ast:
            x1_decoded += ast_idx2word[x1_encoded] + " "

        # decode x2
        for x2_encoded_num in reversed(x2[idx]):
            if x2_encoded_num != 0:
                x2_decoded = comment_idx2word[x2_encoded_num] + " " + x2_decoded

        y_encoded = y[idx]
        y_argmax = np.argmax(y_encoded)
        if y_argmax != 0:
            y_decoded = comment_idx2word[y_argmax]

        print("X1", idx, x1_decoded)
        print("X2", x2_decoded)
        print("Y", y_decoded)
        print("---")


def save_tokenizer(tokenizer, file_path):
    with open(file_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer successfully saved ({})".format(file_path))


def load_tokenizer(file_path):
    with open(file_path, "rb") as handle:
        tokenizer = pickle.load(handle)
        print("Tokenizer loaded ({})".format(file_path))
        return tokenizer


def evaluate(ast_in, ast_tokenizer, comment_tokenizer, max_seq_len, model):
    result = "<start>"

    signature_seq = ast_tokenizer.texts_to_sequences([ast_in])[0]
    signature_seq = pad_sequences(
        [signature_seq], maxlen=max_seq_len, padding="post", truncating="post"
    )
    signature_seq = np.array(signature_seq)

    for i in range(max_seq_len):
        body_seq = comment_tokenizer.texts_to_sequences([result])[0]
        body_seq = pad_sequences(
            [body_seq], maxlen=max_seq_len, padding="post", truncating="post"
        )
        body_seq = np.array(body_seq)

        # predict next token
        y_hat = model.predict([signature_seq, body_seq], verbose=0)
        y_hat = np.argmax(y_hat)

        # map idx to word
        word = comment_tokenizer.index_word[y_hat]

        # append as input for generating the next token
        result += " " + word

        if word is None or word == "<end>":
            break

    return result

class ConditionalScope:
    def __init__(self, scope_factory, enabled = True):
        self.scope = scope_factory() if enabled else None
    
    def __enter__(self):
        if self.scope:
            self.scope.__enter__()

    def __exit__(self, type, value, traceback):
        if self.scope:
            self.scope.__exit__(type, value, traceback)
