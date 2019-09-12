import pickle
import sys
from argparse import ArgumentParser
from os import path

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from utils import load_tokenizer, evaluate
from parameters import UNITS, EMBEDDING_DIM

parser = ArgumentParser()

parser.add_argument("-r", "--run", type=str)
parser.add_argument("--ast", type=str, const=True, nargs="?")

args = parser.parse_args()

if not args.run:
    sys.exit("Error: Please specify a run.")

run = args.run

run_dir = "../runs/{}".format(run)

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to train a model first.".format(run)
    )

params = {}
with open(path.join(run_dir, "params.pickle"), "rb") as f:
    params = pickle.load(f)

print(params)

max_seq_len = params.get("max_seq_len")

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

if __name__ == "__main__":
    ast_in = "FunctionDeclaration ( SyntaxList ( ExportKeyword ) FunctionKeyword Identifier OpenParenToken SyntaxList ( Parameter ( Identifier ColonToken TypeReference ( Identifier ) ) ) CloseParenToken Block ( OpenBraceToken SyntaxList ( ReturnStatement ( ReturnKeyword NewExpression ( NewKeyword Identifier OpenParenToken SyntaxList ( Identifier ) CloseParenToken ) SemicolonToken ) ) CloseBraceToken )"

    ast_in = args.ast if args.ast else ast_in

    print("Using run from '{}'\n".format(run_dir))

    result = evaluate(
        ast_in,
        ast_tokenizer=ast_tokenizer,
        comment_tokenizer=comment_tokenizer,
        max_seq_len=max_seq_len,
        model=model,
    )

    print("AST: {}\n".format(ast_in))
    print("Predicted Comment: {}".format(result))
