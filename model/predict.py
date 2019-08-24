import pickle
import sys
from argparse import ArgumentParser
from os import path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import load_tokenizer

parser = ArgumentParser()

parser.add_argument("-r", "--run", type=str)
parser.add_argument("--ast", type=str, const=True, nargs="?")
parser.add_argument("--maxlen", type=int, const=True, nargs="?", default=500)

args = parser.parse_args()

if not args.run:
    sys.exit("Error: Please specify a run.")

run = args.run

run_dir = "../runs/{}".format(run)

if not path.exists(run_dir):
    sys.exit(
        "Error: Run {} does not exist. Make sure to train embeddings and a model first.".format(
            run
        )
    )

model_file = path.join(run_dir, "model.h5")

if not path.exists(model_file):
    sys.exit("Error: Cannot find '{}'.".format(model_file))

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


def predict(ast_input):
    in_text = "<start>"
    max_tokens = 500
    max_seq_len = args.maxlen

    ast_seq = ast_tokenizer.texts_to_sequences([ast_input])[0]
    ast_seq = pad_sequences(
        [ast_seq], maxlen=max_seq_len, padding="post", truncating="post"
    )[0]
    ast_seq = np.array([ast_seq])

    for _ in range(max_tokens):
        comment_seq = comment_tokenizer.texts_to_sequences([in_text])[0]
        comment_seq = pad_sequences(
            [comment_seq], maxlen=max_seq_len, padding="post", truncating="post"
        )[0]
        comment_seq = np.array([comment_seq])

        # predict next token
        y_hat = model.predict([ast_seq, comment_seq], verbose=0)
        y_hat = np.argmax(y_hat)

        # map idx to word
        word = comment_tokenizer.index_word[y_hat]

        # append as input for generating the next token
        in_text += " " + word

        print(in_text)

        if word is None or word == "END":
            break

    return in_text


if __name__ == "__main__":
    ast_input = "FunctionDeclaration ( SyntaxList ( ExportKeyword ) FunctionKeyword Identifier OpenParenToken SyntaxList ( Parameter ( Identifier ColonToken TypeReference ( Identifier ) ) ) CloseParenToken Block ( OpenBraceToken SyntaxList ( ReturnStatement ( ReturnKeyword NewExpression ( NewKeyword Identifier OpenParenToken SyntaxList ( Identifier ) CloseParenToken ) SemicolonToken ) ) CloseBraceToken )"

    ast_input = args.ast if args.ast else ast_input

    print("Using '{}'".format(run_dir))

    print(predict(ast_input))
