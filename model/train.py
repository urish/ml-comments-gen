import pickle
import re
import sys
import time
from argparse import ArgumentParser
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Embedding,
    concatenate,
    LSTM,
    BatchNormalization,
    Dropout,
    Input,
    Reshape,
    Dense,
)

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from parameters import EMBEDDING_DIM, MAX_LENGTH, UNITS, VOCAB_SIZE
from utils import evaluate, max_length, prepare_dataset, save_tokenizer, check_encoding, ConditionalScope

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()

parser.add_argument("-r", "--run", nargs="?", type=str, const=True)

parser.add_argument("-e", "--epochs", nargs="?", type=int, const=True, default=150)

parser.add_argument("-wv", "--word-vectors", nargs="?", type=int, const=True)

parser.add_argument("-bs", "--batch-size", nargs="?", type=int, const=True, default=32)

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
batch_size = args.batch_size
epochs = args.epochs
word_vectors = args.word_vectors

run_dir = "../runs/{}".format(run)
dataset_path = path.join(run_dir, "dataset_clean.csv")

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to prepare some data first.".format(
            run
        )
    )


def create_tokenizer(name, num_words=None):
    print("Creating tokenizer for '{}'".format(name))
    print("Using num_words={}".format(num_words))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters="", split=" ", lower=False, oov_token="UNK", num_words=num_words
    )

    return tokenizer


df = pd.read_csv(dataset_path)
n_observations = df.shape[0]

print("Observations: {}".format(n_observations))

asts = df["ast"]
comments = df["comments"]

comment_tokenizer = create_tokenizer("Comments", VOCAB_SIZE)
comment_tokenizer.fit_on_texts(comments)

ast_tokenizer = create_tokenizer("ASTs")
ast_tokenizer.fit_on_texts(asts)

# translate each word to the matching vocabulary index
ast_sequences = ast_tokenizer.texts_to_sequences(asts)
comment_sequences = comment_tokenizer.texts_to_sequences(comments)

x1_train = ast_sequences
x2_train = comment_sequences

print("x1 Train:", len(x1_train))
print("x2 Train:", len(x2_train))

len_comment_word_index = len(comment_tokenizer.word_index) + 1

# add +1 to leave space for sequence paddings
x1_vocab_size = len(ast_tokenizer.word_index) + 1
x2_vocab_size = (
    len_comment_word_index if len_comment_word_index < VOCAB_SIZE else VOCAB_SIZE
)

print("x1_vocab_size:", x1_vocab_size)
print("x2_vocab_size:", x2_vocab_size)

# finalize inputs to the model
x1_train, x2_train, y_train, max_seq_len = prepare_dataset(
    "Train", x1_train, x2_train, MAX_LENGTH, x2_vocab_size
)

print("Observations (Train)", max(len(x1_train), len(x2_train)))
print("Max Sequence Length:", max_seq_len)

if debug:
    check_encoding(
        x1_train,
        x2_train,
        y_train,
        ast_tokenizer.index_word,
        comment_tokenizer.index_word,
    )

def create_tpu_scope():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy.scope()

with ConditionalScope(create_tpu_scope, tpu):
    x1_input = Input(shape=x1_train[0].shape, name="x1_input")
    x1_model = Embedding(
        x1_vocab_size, EMBEDDING_DIM, input_length=max_seq_len, mask_zero=True
    )(x1_input)
    x1_model = LSTM(256, return_sequences=True, name="x1_lstm_1")(x1_model)
    x1_model = BatchNormalization()(x1_model)
    x1_model = Dense(128, activation="relu", name="x1_out_hidden")(x1_model)

    x2_input = Input(shape=x2_train[0].shape, name="x2_input")
    x2_model = Embedding(
        x2_vocab_size, EMBEDDING_DIM, input_length=max_seq_len, mask_zero=True
    )(x2_input)
    x2_model = LSTM(256, return_sequences=True, name="x2_lstm_1")(x2_model)
    x2_model = LSTM(256, return_sequences=True, name="x2_lstm_2")(x2_model)
    x2_model = BatchNormalization()(x2_model)
    x2_model = Dense(128, activation="relu", name="x2_out_hidden")(x2_model)

    # decoder
    decoder = concatenate([x1_model, x2_model])
    decoder = LSTM(512, return_sequences=False, name="decoder_lstm")(decoder)
    decoder_output = Dense(x2_vocab_size, activation="softmax")(decoder)

    # compile model
    model = Model(inputs=[x1_input, x2_input], outputs=decoder_output)
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    # auto-save weights
    checkpoint_path = path.join(run_dir, 'checkpoints/cp-{epoch:05d}.ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(path.dirname(checkpoint_path))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=5)

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        initial_epoch = int(path.splitext(latest_checkpoint)[0].split('cp-')[1])
        print("Restored model from checkpoint {}, epoch {}".format(latest_checkpoint, initial_epoch))
    else:
        initial_epoch = 0
        model.save_weights(checkpoint_path.format(epoch=0))

    print("Buckle up and hold tight! We are about to start the training...")
    model.fit(
        [x1_train, x2_train],
        y_train,
        epochs=epochs,
        initial_epoch=initial_epoch,
#        steps_per_epoch=1434,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[cp_callback],
    )

    # save model
    model.save(run_dir + "/model.h5")
    print("Model successfully saved.")

    print("------------------------------")
    print("|           TEST             |")
    print("------------------------------")

    # test model on single input

    ast_in = np.random.choice(asts.values, 1)[0]

    result = evaluate(
        ast_in,
        ast_tokenizer=ast_tokenizer,
        comment_tokenizer=comment_tokenizer,
        max_seq_len=max_seq_len,
        model=model,
    )

    print("Input: %s\n" % (ast_in))
    print("Predicted Comment: {}\n".format(result))

    # save tokenizer
    save_tokenizer(ast_tokenizer, path.join(run_dir, "ast_tokenizer.pickle"))
    save_tokenizer(comment_tokenizer, path.join(run_dir, "comment_tokenizer.pickle"))

    # save training variables
    params = {
        "max_seq_len": max_seq_len,
        "x1_vocab_size": x1_vocab_size,
        "x2_vocab_size": x2_vocab_size,
        "batch_size": batch_size,
    }

    params_path = path.join(run_dir, "params.pickle")

    with open(params_path, "wb") as f:
        pickle.dump(params, f)
        print("\nParams successfully saved ({})".format(params_path))
