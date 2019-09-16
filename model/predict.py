import pickle
import sys
from argparse import ArgumentParser
from os import path

from tensorflow.keras.models import load_model
from utils import load_tokenizer, predict_comment

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

    max_ast_len = params.get('max_ast_len')
    max_comment_len = params.get('max_comment_len')
    ast_vocab_size = params.get('ast_vocab_size')
    comment_vocab_size = params.get('comment_vocab_size')
    lstm_layer_size = params.get('lstm_layer_size')

    model = load_model(path.join(run_dir, 'model.h5'))
    result = predict_comment(
        ast_in, ast_tokenizer, comment_tokenizer, max_ast_len, max_comment_len, 
        model, ast_vocab_size, comment_vocab_size, lstm_layer_size
    )
    print("AST: {}\n".format(ast_in))
    print("Predicted Comment: {}".format(result))
