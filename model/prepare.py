import gzip
import multiprocessing
import re
import sys
from argparse import ArgumentParser
from math import floor
from os import listdir, makedirs, path
from shutil import copyfile

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from spacy_langdetect import LanguageDetector
from yaspin import yaspin

from visualize_embeddings import tsne_plot

ORTH = spacy.symbols.ORTH


def round_down(num):
    i = str(num)
    divisor_max = 2 if num < 1000 else 3
    divisor_idx = min(len(i), divisor_max)

    divisor = "".join(i[:divisor_idx])

    for i in range(len(divisor), len(i)):
        divisor = divisor + "0"

    divisor = int(divisor)

    return floor(num / divisor) * divisor


def splitForwardSlashes(match):
    slashes = list(match.group(0))
    return " ".join(slashes)


def clean_token(text):
    # normalize
    text = text.lower()

    # replace URLs
    text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "URL", text)

    # custom splitting rule to create space around special code characters
    text = re.sub(r"([.,()<>\[\]{}\"\'`\-$=_;%|&#~^\\])", r" \1 ", text)

    # replace multiple forward slashes
    text = re.sub(r"\/{3,}", lambda m: splitForwardSlashes(m), text)

    # replace tab stops
    text = re.sub(r"\t", "", text)

    # replace newlines with a special "end-of-sequence" token
    text = re.sub(r"\r\n|\r|\n", " <eol> ", text)

    return text


def clean_document(doc):
    """ Clean up comments. Tokenize, lowercase, and remove characters that are not allowed """

    # filter out English comments
    isEnglish = lambda doc: doc._.language["language"] == "en"

    if not isEnglish(doc):
        return nlp.make_doc("")

    # clean each token in document
    text = [token for token in (clean_token(tok.text) for tok in doc) if token != ""]

    text = " ".join(text)

    # adding a start and an end token to the sentence so that
    # the model know when to start and stop predicting
    text = "<start> " + text + " <end>"

    # replace multiple whitespaces with a single whitespace
    text = re.sub(r" {2,}", " ", text)

    return nlp.make_doc(text)


nlp = spacy.load("en_core_web_sm")

# add special cases for the tokenizer
nlp.tokenizer.add_special_case("/**", [{ORTH: "/**"}])
nlp.tokenizer.add_special_case("/*", [{ORTH: "/*"}])
nlp.tokenizer.add_special_case("*/", [{ORTH: "*/"}])
nlp.tokenizer.add_special_case("//", [{ORTH: "//"}])
nlp.tokenizer.add_special_case("<start>", [{ORTH: "<start>"}])
nlp.tokenizer.add_special_case("<eol>", [{ORTH: "<eol>"}])
nlp.tokenizer.add_special_case("<end>", [{ORTH: "<end>"}])

nlp.add_pipe(clean_document, name="cleaner", last=True)
nlp.add_pipe(LanguageDetector(), name="language_detector", before="cleaner")

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()

parser.add_argument(
    "-v", "--visualize", nargs="?", type=boolean, const=True, default=False
)

parser.add_argument(
    "-s", "--save-dataset", nargs="?", type=boolean, const=True, default=True
)

parser.add_argument("-t", "--train", nargs="?", type=boolean, const=True, default=False)

parser.add_argument(
    "-d", "--dataset", nargs="?", type=str, const=True, default="dataset.json"
)

args = parser.parse_args()

saveDataset = args.save_dataset
visualize = args.visualize
train = args.train

data_dir = "../data"
metadata_filename = "metadata.txt"
metadata_path = path.join(data_dir, metadata_filename)
dataset_path = path.join(data_dir, args.dataset)
filename_dataset_clean = "dataset_clean.csv"

if not path.exists(dataset_path):
    sys.exit(
        "Error: Couldn't find '{}'. Make sure to generate a dataset first.".format(
            path.basename(dataset_path)
        )
    )

out_dir = "../runs"

# create output dir
if not path.exists(out_dir):
    makedirs(out_dir)

df = pd.read_json(dataset_path, lines=True)

# create run dir
run_dir = ""

# --- Hyper-Parameters ---

# more dimensions mean more computationally expensive,
# but also more accurate. 300 is a decent compromise
num_features = 300

# minimum count of words to consider when training the model
min_word_count = 3

# run training in parallel, more workers = faster training
num_workers = multiprocessing.cpu_count()

# size of the sliding window (number of words around the target window)
window_size = 7

# determines how often do we want to look at the same word
downsampling = 1e-3

# used to pick what part of the text we look at
seed = 1

# default is 5, we keep the default because increasing the number
# of epochs dramatically increases the training time, but also gives
# better results.
epochs = 5

# ------------------------


@yaspin(text="Cleaning comments...")
def clean_comments(comments):
    comments = comments.apply(lambda c: nlp(c))
    comments = comments.apply(lambda doc: doc if doc.text != "" else np.nan)
    return comments


@yaspin(text="Dumping dataset...")
def dump_dataset(df):
    df.to_csv(path.join(run_dir, filename_dataset_clean), header=True)


@yaspin(text="Training Word2vec...")
def train_word2vec(
    sentences,
    sg=1,
    size=num_features,
    min_count=min_word_count,
    seed=seed,
    window=window_size,
    sample=downsampling,
    iter=epochs,
):
    model = Word2Vec(
        sentences,
        sg=sg,
        size=size,
        min_count=min_count,
        seed=seed,
        window=window,
        workers=num_workers,
        sample=sample,
        iter=iter,
    )

    return model


@yaspin(text="Saving model...")
def save_model(model, filename):
    model_path = path.join(run_dir, filename)
    model.save(model_path)


@yaspin(text="Plotting word embeddings...")
def plot_embeddings(model, df, filename):
    tsne_plot(model, df, filename)

df["comments_orig"] = df["comments"]
df["comments"] = clean_comments(df["comments"])

# remove corrupted rows (mostly comments that are written in languages other than English)
df = df.dropna()

n_observations = df.shape[0]
n_observations_r = round_down(n_observations)

run_dir = path.join(out_dir, str(n_observations_r))

if not path.exists(run_dir):
    makedirs(run_dir)

print("Observations: {}".format(n_observations))

comments = df["comments"].map(lambda doc: [token.text for token in doc])

if train:
    model_comments = train_word2vec(comments)
    save_model(model_comments, "word2vec_comments.model")
    print("Size Vocabulary (Comments):", len(model_comments.wv.vocab))

    if visualize:
        plot_embeddings(model_comments, df, path.join(run_dir, "word2vec_comments.png"))

asts = df["ast"]
asts = asts.map(lambda ast: [token for token in ast.split(" ")])

if train:
    model_asts = train_word2vec(asts, min_count=1)
    save_model(model_asts, "word2vec_asts.model")
    print("Size Vocabulary (ASTs):", len(model_asts.wv.vocab))

    if visualize:
        plot_embeddings(model_asts, df, path.join(run_dir, "word2vec_asts.png"))

dataset_clean = df[["ast", "comments", "comments_orig"]]

if saveDataset:
    dump_dataset(dataset_clean)

print("Copying metadata")
copyfile(metadata_path, path.join(run_dir, metadata_filename))

print("Done!")
