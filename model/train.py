import re
import time
import sys
from argparse import ArgumentParser
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import max_length, prepare_dataset, check_encoding, create_embedding_matrix
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()

parser.add_argument("-r", "--run", nargs="?", type=str, const=True)

parser.add_argument("-e", "--epochs", nargs="?", type=int, const=True, default=100)

parser.add_argument("-bs", "--batch-size", nargs="?", type=int, const=True, default=64)

parser.add_argument(
    "-v", "--visualize", nargs="?", type=boolean, const=True, default=False
)

parser.add_argument("-d", "--debug", nargs="?", type=boolean, const=True, default=False)

parser.add_argument("-tpu", nargs="?", type=boolean, const=True, default=False)

args = parser.parse_args()

if not args.run:
    sys.exit("Error: Please specify a run.")

visualize = args.visualize
run = args.run
debug = args.debug
tpu = args.tpu

run_dir = "../runs/{}".format(run)
dataset_path = path.join(run_dir, "dataset_clean.csv")

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to train embeddings first.".format(run)
    )


def load_model(name):
    return Word2Vec.load(path.join(run_dir, "word2vec_{}.model".format(name)))


def create_tokenizer(w2v_model, name):
    print("Creating tokenizer for '{}'".format(name))

    model = w2v_model.wv
    vocab = model.vocab

    print("Size Vocabulary {} (Raw):".format(name), len(vocab))

    # for both we have to shift the index by +1, because by default Word2Vec is 0-based
    # IMPORTANT: We also have to shift the vector matrix that we use for the embedding layer
    index2word = {i + 1: w for i, w in enumerate(model.index2word)}
    word2idx = {word: vocab[word].index + 1 for i, word in enumerate(vocab.keys())}

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters="", split=" ", lower=False, oov_token="UNK"
    )

    tokenizer.index_word = index2word
    tokenizer.word_index = word2idx

    # add out-of-vocabulary token
    oov_idx = len(vocab) + 1
    tokenizer.index_word[oov_idx] = tokenizer.oov_token
    tokenizer.word_index[tokenizer.oov_token] = oov_idx

    return tokenizer


df = pd.read_csv(dataset_path)
n_observations = df.shape[0]

print("Observations: {}".format(n_observations))

word2vec_comments = load_model("comments")
word2vec_asts = load_model("asts")

comment_tokenizer = create_tokenizer(word2vec_comments, "Comments")
ast_tokenizer = create_tokenizer(word2vec_asts, "ASTs")

asts = df["ast"]
comments = df["comments"]

# translate each word to the matching vocabulary index
ast_sequences = ast_tokenizer.texts_to_sequences(asts)
comment_sequences = comment_tokenizer.texts_to_sequences(comments)

# train, test split
seed = 1
test_size = 0.33
x1_train, x1_test, x2_train, x2_test = train_test_split(
    ast_sequences, comment_sequences, test_size=test_size, random_state=seed
)

print("x1 Train:", len(x1_train))
print("x1 Test:", len(x1_test))
print("x2 Train:", len(x2_train))
print("x2 Test:", len(x2_test))

# add +1 to leave space for sequence paddings
x1_vocab_size = len(ast_tokenizer.word_index) + 1
x2_vocab_size = len(comment_tokenizer.word_index) + 1

# NOTE: try to increase max length and see if it improves the model performance
MAX_LENGTH = 500

# finalize inputs to the model
x1_train, x2_train, y_train = prepare_dataset(
    x1_train, x2_train, MAX_LENGTH, x2_vocab_size
)

x1_test, x2_test, y_test = prepare_dataset(x1_test, x2_test, MAX_LENGTH, x2_vocab_size)

max_length = max(x1_train.shape[1], x2_train.shape[1])

if debug:
    check_encoding(
        x1_train,
        x2_train,
        y_train,
        ast_tokenizer.index_word,
        comment_tokenizer.index_word,
    )

# # --- Model ---

EMBEDDING_DIM = 300

comment_embedding_matrix = create_embedding_matrix(
    EMBEDDING_DIM, comment_tokenizer.word_index, word2vec_comments.wv
)

ast_embedding_matrix = create_embedding_matrix(
    EMBEDDING_DIM, ast_tokenizer.word_index, word2vec_asts.wv
)

# --- Encoder --- START ---

# --- X1 ---

x1_input = tf.keras.layers.Input(shape=x1_train[0].shape, name="x1_input")

x1_model = tf.keras.layers.Embedding(
    x1_vocab_size,
    EMBEDDING_DIM,
    weights=[ast_embedding_matrix],
    input_length=max_length,
    trainable=False,
)(x1_input)

x1_model = tf.keras.layers.GRU(1024, return_sequences=True, name="x1_gru_1")(x1_model)
x1_model = tf.keras.layers.GRU(512, return_sequences=True, name="x1_gru_2")(x1_model)
x1_model = tf.keras.layers.Dense(128, activation="relu", name="x1_out_hidden")(x1_model)

# --- X2 ---

x2_input = tf.keras.layers.Input(shape=x2_train[0].shape, name="x2_input")

x2_model = tf.keras.layers.Embedding(
    x2_vocab_size,
    EMBEDDING_DIM,
    weights=[comment_embedding_matrix],
    input_length=max_length,
    trainable=False,
)(x2_input)

x2_model = tf.keras.layers.GRU(1024, return_sequences=True, name="x2_gru_1")(x2_model)
x2_model = tf.keras.layers.GRU(512, return_sequences=True, name="x2_gru_2")(x2_model)
x2_model = tf.keras.layers.Dense(128, activation="relu", name="x2_out_hidden")(x2_model)

# Encoder --- END ---

# Decoder --- START ---

decoder = tf.keras.layers.concatenate([x1_model, x2_model])
decoder = tf.keras.layers.GRU(1024, return_sequences=False, name="decoder_gru")(decoder)
decoder_output = tf.keras.layers.Dense(x2_vocab_size, activation="softmax")(decoder)

# Decoder --- END ---

model = tf.keras.models.Model(inputs=[x1_input, x2_input], outputs=decoder_output)

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

# Training

print("Buckle up and hold tight! We are about to start the training...")

validation_data = ([x1_test, x2_test], y_test)

# --- Hyper-Parameters ---

batch_size = args.batch_size
n_epochs = args.epochs

print("Batch Size:", batch_size)
print("Epochs:", n_epochs)

# ------------------------

print("Shape X1 (AST):", x1_train.shape)
print("Shape X2 (Comment):", x2_train.shape)
print("Shape Y (Next Token):", y_train.shape)

# if tpu:
#     model = tf.contrib.tpu.keras_to_tpu_model(
#         model,
#         strategy=tf.contrib.tpu.TPUDistributionStrategy(
#             tf.contrib.cluster_resolver.TPUClusterResolver()
#         ),
#     )

model.fit(
    [x1_train, x2_train],
    y_train,
    # validation_data=validation_data,
    epochs=n_epochs,
    batch_size=batch_size,
    shuffle=False,
)

# save model
model.save(run_dir + "/model.h5")
print("Model successfully saved.")
