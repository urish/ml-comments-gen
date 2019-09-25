import pickle
import sys
from argparse import ArgumentParser
from os import path
import json
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow_datasets.core.features.text.subword_text_encoder import SubwordTextEncoder

from utils import ConditionalScope, save_tokenizer, predict_comment

from parameters import (
    EMBEDDING_DIM,
    MAX_AST_LEN,
    MAX_COMMENT_LEN,
    UNITS,
    VOCAB_SIZE,
    LSTM_LAYER_SIZE,
)

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

parser.add_argument("-c", "--checkpoints", nargs="?", type=boolean, const=True, default=True)

parser.add_argument("--checkpoint-root", nargs="?", type=str, const=True)

parser.add_argument("-tb", "--tensorboard", nargs="?", type=boolean, const=True, default=False)

parser.add_argument("--steps-per-epoch", nargs="?", type=int, const=True)

parser.add_argument("--skip-test", nargs="?", type=boolean, const=True, default=False)

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
checkpoints = args.checkpoints
tensorboard = args.tensorboard
steps_per_epoch = args.steps_per_epoch
checkpoint_root = args.checkpoint_root
skip_test = args.skip_test

run_dir = "../runs/{}".format(run)
dataset_path = path.join(run_dir, "dataset_clean.csv")

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to prepare some data first.".format(
            run
        )
    )


def create_tokenizer(name, num_words=None, char_level=False):
    print("Creating tokenizer for '{}'".format(name))
    print("Using num_words={}".format(num_words))
    return tf.keras.preprocessing.text.Tokenizer(
        filters="", split=" ", lower=False, oov_token="UNK", num_words=num_words,
        char_level=char_level
    )


df = pd.read_csv(dataset_path)
n_observations = df.shape[0]

print("Observations: {}".format(n_observations))

asts = df["ast"]
comments = df["comments_orig"]
comment_tokenizer = SubwordTextEncoder.build_from_corpus(comments, target_vocab_size=VOCAB_SIZE)

ast_tokenizer = create_tokenizer("ASTs")
ast_tokenizer.fit_on_texts(asts)

# we add two to account for padding and an eof token
ast_vocab_size = len(ast_tokenizer.word_index) + 2
comment_start_token = comment_tokenizer.vocab_size
comment_end_token = comment_start_token + 1
comment_vocab_size = comment_end_token + 1
print("Vocabulary size: ast={}, comments={}".format(ast_vocab_size, comment_vocab_size))

# translate each word to the matching vocabulary index
ast_sequences = ast_tokenizer.texts_to_sequences(asts)
comment_sequences = [[comment_start_token] + comment_tokenizer.encode(comment) + [comment_end_token] for comment in comments]

encoder_input_data = np.zeros(
    (len(ast_sequences), MAX_AST_LEN), dtype="float32"
)

decoder_input_data = np.zeros(
    (len(comment_sequences), MAX_COMMENT_LEN), dtype="float32"
)

decoder_target_data = np.zeros(
    (len(comment_sequences), MAX_COMMENT_LEN, comment_vocab_size), dtype="float32"
)

ast_eof_token = ast_vocab_size - 1

for i, (input_text, target_text) in enumerate(zip(ast_sequences, comment_sequences)):
    for t, token in enumerate(input_text):
        encoder_input_data[i, t] = token

    encoder_input_data[i, t + 1:] = ast_eof_token

    for t, token in enumerate(target_text):
        if t >= MAX_COMMENT_LEN:
            continue

        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = token

        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, token] = 1.0

    decoder_input_data[i, t+1:] = comment_end_token
    decoder_target_data[i, t:, comment_end_token] = 1.0


def create_tpu_scope():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy.scope()


# Model code based on https://keras.io/examples/lstm_seq2seq/

with ConditionalScope(create_tpu_scope, tpu):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    encoder_embeddings = Embedding(ast_vocab_size, EMBEDDING_DIM)(encoder_inputs)
    encoder = LSTM(LSTM_LAYER_SIZE, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    decoder_embeddings = Embedding(comment_vocab_size, EMBEDDING_DIM)(decoder_inputs)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(LSTM_LAYER_SIZE, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)
    decoder_dense = Dense(comment_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    fit_callbacks = []

    # auto-save weights
    initial_epoch = 0

    if checkpoints:
        checkpoint_path = path.join(checkpoint_root or run_dir, "checkpoints/cp-{epoch:05d}.ckpt")
        latest_checkpoint = tf.train.latest_checkpoint(path.dirname(checkpoint_path))

        fit_callbacks.append(ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True, period=5
        ))

        if latest_checkpoint:
            model.load_weights(latest_checkpoint)
            initial_epoch = int(path.splitext(latest_checkpoint)[0].split("cp-")[1])
            print(
                "Restored model from checkpoint {}, epoch {}".format(
                    latest_checkpoint, initial_epoch
                )
            )
        else:
            model.save_weights(checkpoint_path.format(epoch=0))

    # Tensorboard logs
    if tensorboard:
        fit_callbacks.append(TensorBoard(log_dir=run_dir))

    print("Buckle up and hold tight! We are about to start the training...")
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        callbacks=fit_callbacks,
        steps_per_epoch=steps_per_epoch,
    )

    # save model
    model.save(run_dir + "/model.h5")
    print("Model successfully saved.")

    # save tokenizer
    save_tokenizer(ast_tokenizer, path.join(run_dir, "ast_tokenizer.pickle"))
    save_tokenizer(comment_tokenizer, path.join(run_dir, "comment_tokenizer.pickle"))

    # save training variables
    params = {
        "max_ast_len": MAX_AST_LEN,
        "max_comment_len": MAX_COMMENT_LEN,
        "ast_vocab_size": ast_vocab_size,
        "lstm_size": LSTM_LAYER_SIZE,
        "comment_vocab_size": comment_vocab_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "comment_start_token": comment_start_token,
        "comment_end_token": comment_end_token,
    }

    params_path = path.join(run_dir, "params.pickle")

    with open(params_path, "wb") as f:
        pickle.dump(params, f)
        print("\nParams successfully saved ({})".format(params_path))

    # Save training history report
    history_path = path.join(run_dir, "training.json")
    with open(history_path, "w") as f:
        json.dump({
            "history": [
                {"epoch": epoch, "loss": loss.item(), "acc": acc.item()} for (epoch, loss, acc) in zip(history.epoch, history.history['loss'], history.history['acc'])
            ]
        }, f, indent=2)

    if not skip_test:
        print("------------------------------")
        print("|           TEST             |")
        print("------------------------------")

        # test model on single input

        ast_in = np.random.choice(asts.values, 1)[0]

        result = predict_comment(
            ast_in,
            ast_tokenizer,
            comment_tokenizer,
            MAX_AST_LEN,
            MAX_COMMENT_LEN,
            model,
            ast_vocab_size,
            comment_vocab_size,
            LSTM_LAYER_SIZE,
            comment_start_token,
            comment_end_token,
        )

        print("Input: %s\n" % (ast_in))
        print("Predicted Comment: {}\n".format(result))
