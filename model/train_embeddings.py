import gzip
import multiprocessing
import re
import sys
from argparse import ArgumentParser
from os import path

import numpy as np
import pandas as pd
import spacy
from yaspin import yaspin
from gensim.models import Word2Vec
from spacy_langdetect import LanguageDetector

from visualize_embeddings import tsne_plot

ORTH = spacy.symbols.ORTH


def clean_token(text):
    # normalize
    text = text.lower()

    # replace URLs
    text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "URL", text)

    # custom splitting rule to create space around special code characters
    text = re.sub(r"([.,()<>\[\]{}\"\'`\-$=_;%|&#~^])", r" \1 ", text)

    # replace tab stops
    text = re.sub(r"\t", "", text)

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
nlp.tokenizer.add_special_case("<end>", [{ORTH: "<end>"}])

nlp.add_pipe(clean_document, name="cleaner", last=True)
nlp.add_pipe(LanguageDetector(), name="language_detector", before="cleaner")

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()

parser.add_argument(
    "-v", "--visualize", nargs="?", type=boolean, const=True, default=False
)

parser.add_argument(
    "-dd", "--dump-dataset", nargs="?", type=boolean, const=True, default=True
)

parser.add_argument(
    "-d", "--dataset", nargs="?", type=str, const=True, default="dataset.json"
)

args = parser.parse_args()

data_dir = "../data"
dataset_path = path.join(data_dir, args.dataset)
filename_word2vec_model = "word2vec.model"
filename_dataset_clean = "dataset_clean.csv"

if not path.exists(dataset_path):
    sys.exit(
        "Error: Couldn't find '{}'. Make sure to generate a dataset first.".format(
            path.basename(dataset_path)
        )
    )

df = pd.read_json(dataset_path, lines=True)

n_observations = df.shape[0]

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


@yaspin(text="Cleaning dataset...")
def clean_dataset(df):
    df = df.apply(lambda c: nlp(c))
    df = df.apply(lambda doc: doc if doc.text != "" else np.nan)
    df = df.dropna()
    return df


@yaspin(text="Dumping dataset...")
def dump_dataset(df):
    filename = get_filename(filename_dataset_clean)
    df.to_csv(path.join(data_dir, filename), header=True)


def get_filename(name):
    segments = path.splitext(name)
    filename = segments[0]
    ext = segments[1]

    return "{}_{}_{}{}".format(filename, n_observations, epochs, ext)


@yaspin(text="Preparing model inputs...")
def prepare_word2vec_inputs(documents):
    # create list (of comments) of lists (tokens)
    # e.g. [['this', 'is', 'a', 'comment'], ['second', 'comment'], ['another', 'comment']]
    return documents.map(lambda doc: [token.text for token in doc])


@yaspin(text="Training Word2vec...")
def train_word2vec(sentences):
    model = Word2Vec(
        sentences,
        sg=1,
        size=num_features,
        min_count=min_word_count,
        seed=seed,
        window=window_size,
        workers=num_workers,
        sample=downsampling,
        iter=epochs,
    )

    return model


@yaspin(text="Saving model...")
def save_model(model):
    model_path = path.join(data_dir, get_filename(filename_word2vec_model))
    model.save(model_path)


@yaspin(text="Plotting word embeddings...")
def plot_embeddings(model, df, filename):
    tsne_plot(model, df, filename)


def train_embeddings(dump_data=None, plot=None, retrain=False):
    visualize = plot if plot is not None else args.visualize
    dumpDataset = dump_data if dump_data is not None else args.dump_dataset

    clean_dataset_path = path.join(data_dir, get_filename(filename_dataset_clean))
    model_path = path.join(data_dir, get_filename(filename_word2vec_model))

    print("Observations: {}".format(n_observations))

    if path.exists(clean_dataset_path) and path.exists(model_path) and retrain == False:
        print(
            "Skipping training and reusing existing model. Using '{}'.".format(
                path.basename(model_path)
            )
        )

        model = Word2Vec.load(model_path)

        print("Size Vocabulary:", len(model.wv.vocab))

        dataset = pd.read_csv(clean_dataset_path)
        return model, dataset

    comments = df["comments"]

    comments = clean_dataset(comments)
    df["comments"] = comments

    dataset_clean = df[["ast", "comments"]]

    comments = prepare_word2vec_inputs(comments)
    model = train_word2vec(comments)

    if dumpDataset:
        dump_dataset(dataset_clean)

    vocab = list(model.wv.vocab)
    print("Size Vocabulary:", len(vocab))

    if visualize:
        filename = get_filename("word2vec.png")
        plot_embeddings(model, df, filename)

    # save model
    save_model(model)

    print("Done!")

    return model, dataset_clean


if __name__ == "__main__":
    train_embeddings()
