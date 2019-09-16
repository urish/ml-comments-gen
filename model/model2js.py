import pickle
import sys
from argparse import ArgumentParser
from os import path
import json

from tensorflow.keras.models import load_model
import tensorflowjs as tfjs
from utils import load_tokenizer
from parameters import UNITS, EMBEDDING_DIM

parser = ArgumentParser()

parser.add_argument("-r", "--run", type=str)

args = parser.parse_args()

if not args.run:
    sys.exit("Error: Please specify a run.")

run = args.run

run_dir = "../runs/{}".format(run)
jsmodel_dir = path.join(run_dir, "tfjsmodel")

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to train a model first.".format(run)
    )

params = {}
with open(path.join(run_dir, "params.pickle"), "rb") as f:
    params = pickle.load(f)

model_file = path.join(run_dir, "model.h5")
model = load_model(model_file)
print("Model '{}' loaded.".format(model_file))

ast_tokenizer_file = path.join(run_dir, "ast_tokenizer.pickle")
comment_tokenizer_file = path.join(run_dir, "comment_tokenizer.pickle")

if not path.exists(ast_tokenizer_file):
    sys.exit("Error: Cannot load '{}'.".format(ast_tokenizer_file))

if not path.exists(comment_tokenizer_file):
    sys.exit("Error: Cannot load '{}'.".format(comment_tokenizer_file))

ast_tokenizer = load_tokenizer(ast_tokenizer_file)
comment_tokenizer = load_tokenizer(comment_tokenizer_file)

tfjs.converters.save_keras_model(model, jsmodel_dir)

with open(path.join(jsmodel_dir, "tokenizers.json"), "w") as outfile:
    json.dump(
        {
            "params": params,
            "ast": ast_tokenizer.word_index,
            "comments": comment_tokenizer.index_word,
        },
        outfile,
    )

print("Model successfully exported")
