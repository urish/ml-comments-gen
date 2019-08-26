import pickle
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from parameters import UNITS, MAX_LENGTH


def max_length(seqs, max_seq_len=MAX_LENGTH):
    length = max(len(s) for s in seqs)
    return length if length < max_seq_len else max_seq_len


def prepare_dataset(input_raw, target_raw, max_seq_len, name):
    input_maxlen = max_length(input_raw, max_seq_len)
    target_maxlen = max_length(target_raw, max_seq_len)

    print("Preparing {} Dataset".format(name))

    input_tensor = pad_sequences(
        input_raw, maxlen=input_maxlen, padding="post", truncating="post"
    )

    target_tensor = pad_sequences(
        target_raw, maxlen=target_maxlen, padding="post", truncating="post"
    )

    return input_tensor, input_maxlen, target_tensor, target_maxlen


def create_embedding_matrix(dim, word_index, word_vectors):
    max_num_words = len(word_index) + 1

    # we initialize the matrix with zeros
    embedding_matrix = np.zeros((max_num_words, dim))

    for word, i in word_index.items():
        if i >= max_num_words:
            continue

        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except:
            pass

    return embedding_matrix


def save_tokenizer(tokenizer, file_path):
    with open(file_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer successfully saved ({})".format(file_path))


def load_tokenizer(file_path):
    with open(file_path, "rb") as handle:
        tokenizer = pickle.load(handle)
        print("Tokenizer loaded ({})".format(file_path))
        return tokenizer


def evaluate(
    ast_in,
    ast_tokenizer,
    comment_tokenizer,
    max_length_input,
    max_length_target,
    encoder,
    decoder,
    out_dir=None,
    plot=False,
):
    attention_plot = np.zeros((max_length_target, max_length_input))

    input_seq = ast_tokenizer.texts_to_sequences([ast_in])

    input_seq = tf.keras.preprocessing.sequence.pad_sequences(
        input_seq, maxlen=max_length_input, padding="post", truncating="post"
    )

    inputs = tf.convert_to_tensor(input_seq)

    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([comment_tokenizer.word_index["<start>"]], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += comment_tokenizer.index_word[predicted_id] + " "

        if comment_tokenizer.index_word[predicted_id] == "<end>":
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    if plot and out_dir and path.exists(out_dir):
        attention_plot = attention_plot[
            : len(result.split(" ")), : len(ast_in.split(" "))
        ]
        plot_attention(out_dir, attention_plot, ast_in.split(" "), result.split(" "))

    return result


def plot_attention(out_dir, attention, ast_in, predicted_comment):
    print("Plotting attention...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + ast_in, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_comment, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(path.join(out_dir, "attention.png"))
    print("Attention saved to '{}'".format(out_dir))

