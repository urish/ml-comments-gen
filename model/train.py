import pickle
import re
import sys
import time
from argparse import ArgumentParser
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from model import BahdanauAttention, Decoder, Encoder
from parameters import EMBEDDING_DIM, MAX_LENGTH, UNITS, VOCAB_SIZE
from utils import evaluate, max_length, prepare_dataset, save_tokenizer

# TF GPU Config
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()

parser.add_argument("-r", "--run", nargs="?", type=str, const=True)

parser.add_argument("-e", "--epochs", nargs="?", type=int, const=True, default=150)

parser.add_argument("-wv", "--word-vectors", nargs="?", type=int, const=True)

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


def load_w2v_model(name):
    w2v_dir = run_dir if not word_vectors else "../runs/{}".format(word_vectors)

    if word_vectors:
        print("Using embeddings from '{}'".format(w2v_dir))

    return Word2Vec.load(path.join(w2v_dir, "word2vec_{}.model".format(name)))


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

# train, test split
seed = 1
test_size = 0.33
# x1_train, x1_test, x2_train, x2_test = train_test_split(
#     ast_sequences, comment_sequences, test_size=test_size, random_state=seed
# )

x1_train = ast_sequences
x2_train = comment_sequences

print("x1 Train:", len(x1_train))
print("x2 Train:", len(x2_train))
# print("x1 Test:", len(x1_test))
# print("x2 Test:", len(x2_test))

len_comment_word_index = len(comment_tokenizer.word_index) + 1

# add +1 to leave space for sequence paddings
x1_vocab_size = len(ast_tokenizer.word_index) + 1
x2_vocab_size = (
    len_comment_word_index if len_comment_word_index < VOCAB_SIZE else VOCAB_SIZE
)

print("x1_vocab_size:", x1_vocab_size)
print("x2_vocab_size:", x2_vocab_size)

# finalize inputs to the model
input_tensor, max_length_input, target_tensor, max_length_target = prepare_dataset(
    x1_train, x2_train, MAX_LENGTH, "Train"
)

print("Max Sequence Input:", max_length_input)
print("Max Sequence Target:", max_length_target)

# x1_test, x2_test, y_test = prepare_dataset(
#     x1_test, x2_test, MAX_LENGTH, x2_vocab_size, "Test"
# )


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def accuracy_function(real, pred):
    return accuracy.update_state(real, pred)


BUFFER_SIZE = len(input_tensor)

BATCH_SIZE = (
    args.batch_size if len(input_tensor) >= args.batch_size else len(input_tensor)
)

encoder = Encoder(x1_vocab_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
decoder = Decoder(x2_vocab_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

# Create TF.dataset

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(
    BUFFER_SIZE
)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# get sample batch for test outputs
example_encoder_batch, example_target_batch = next(iter(dataset))

print("Sample Batch Shapes:", example_encoder_batch.shape, example_target_batch.shape)

if debug:
    # test encoder output
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_encoder_batch, sample_hidden)

    print(
        "Encoder output shape: (batch size, sequence length, units) {}".format(
            sample_output.shape
        )
    )

    print(
        "Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape)
    )

    # test attention
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print(
        "Attention result shape: (batch size, units) {}".format(attention_result.shape)
    )
    print(
        "Attention weights shape: (batch_size, sequence_length, 1) {}".format(
            attention_weights.shape
        )
    )

    # test decoder output
    sample_decoder_output, _, _ = decoder(
        tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output
    )

    print(
        "Decoder output shape: (batch_size, vocab size) {}".format(
            sample_decoder_output.shape
        )
    )

# --- Training ---


@tf.function
def train_step(encoder_input, target, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(encoder_input, enc_hidden)

        dec_hidden = enc_hidden

        decoder_input = tf.expand_dims(
            [comment_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1
        )

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(decoder_input, dec_hidden, enc_output)

            correct_label = target[:, t]
            loss += loss_function(correct_label, predictions)
            accuracy_function(correct_label, predictions)

            # using teacher forcing
            decoder_input = tf.expand_dims(correct_label, 1)

    batch_loss = loss / int(target.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


steps_per_epoch = len(input_tensor) // BATCH_SIZE

checkpoint_dir = run_dir
checkpoint_prefix = path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

print("Steps per Epoch:", steps_per_epoch)

for epoch in range(epochs):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(
                "Epoch {}/{} Batch {} Loss {:.4f}".format(
                    epoch + 1, epochs, batch, batch_loss.numpy()
                )
            )

    print(
        "Epoch {}/{} Loss {:.4f} Accuracy {:.3f}%".format(
            epoch + 1, epochs, total_loss / BATCH_SIZE, accuracy.result().numpy()
        )
    )

    print("Time taken for 1 epoch {:.3f} sec\n".format(time.time() - start))

# save weights
encoder_path = path.join(run_dir, "encoder")
encoder.save_weights(encoder_path)
print("Encoder weights saved ({})".format(encoder_path))

decoder_path = path.join(run_dir, "decoder")
decoder.save_weights(decoder_path)
print("Decoder weights saved ({})".format(decoder_path))

print("------------------------------")
print("|           TEST             |")
print("------------------------------")

# test model on single input

ast_in = np.random.choice(asts.values, 1)[0]

result = evaluate(
    ast_in,
    ast_tokenizer=ast_tokenizer,
    comment_tokenizer=comment_tokenizer,
    max_length_input=max_length_input,
    max_length_target=max_length_target,
    encoder=encoder,
    decoder=decoder,
    out_dir=run_dir,
    plot=True,
)

print("Input: %s\n" % (ast_in))
print("Predicted Comment: {}\n".format(result))

# save tokenizer
save_tokenizer(ast_tokenizer, path.join(run_dir, "ast_tokenizer.pickle"))
save_tokenizer(comment_tokenizer, path.join(run_dir, "comment_tokenizer.pickle"))

# save training variables
params = {
    "max_length_input": max_length_input,
    "max_length_target": max_length_target,
    "x1_vocab_size": x1_vocab_size,
    "x2_vocab_size": x2_vocab_size,
    "batch_size": BATCH_SIZE,
}

params_path = path.join(run_dir, "params.pickle")

with open(params_path, "wb") as f:
    pickle.dump(params, f)
    print("\nParams successfully saved ({})".format(params_path))
