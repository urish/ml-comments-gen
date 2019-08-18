import gzip
from os import path
from gensim.models import Word2Vec
import logging
import pandas as pd
import re
import nltk
import multiprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from visualize_embeddings import tsne_plot
from argparse import ArgumentParser

boolean = lambda x: (str(x).lower() == "true")

parser = ArgumentParser()
parser.add_argument(
    "-v", "--visualize", nargs="?", type=boolean, const=True, default=False
)

args = parser.parse_args()

visualize = args.visualize

nltk.download("stopwords")
nltk.download("punkt")

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

dataset_dir = "../data/dataset.json"

if not path.exists(dataset_dir):
    raise Exception("'{}' does not exist.".format(dataset_dir))

df = pd.read_json(dataset_dir, lines=True)

logging.info("#Observations: {}".format(df.shape[0]))

comments = df["comments"]

# --- Data Preprocessing ---

# normalize
comments = comments.str.lower()

# remove non-alphabetic words
comments = comments.map(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", x))

# remove special characters
comments = comments.map(lambda x: re.sub(r"[!@#$+%*:()'-]", " ", x))

# remove stop words
stop_words = stopwords.words("english")
comments = comments.map(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

# tokenize each comment, creating a list (of comments) of lists (tokens)
# e.g. [['this', 'is', 'a', 'comment'], ['second', 'comment'], ['another', 'comment']]
comments = comments.map(lambda x: word_tokenize(x))

# --- Word2Vec Training ---

# more dimensions mean more computationally expensive,
# but also more accurate. 300 is a decent compromise
num_features = 300

# minimum count of words to consider when training the model
min_word_count = 5

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

# --- Trainig ---

model = Word2Vec(
    comments,
    sg=1,
    size=num_features,
    min_count=min_word_count,
    window=window_size,
    workers=num_workers,
    sample=downsampling,
    iter=epochs,
)

# print vocabulary size (for orientation purpose)
vocab = list(model.wv.vocab)
logging.info("Size Vocabulary: {}".format(len(vocab)))

if visualize:
    logging.info("Plotting Word2Vec Embeddings")
    tsne_plot(model, df, epochs)

# save model
model.save("embeddings_word2vec.model")
