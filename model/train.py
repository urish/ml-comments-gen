import re
import time
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from train_embeddings import train_embeddings

word2vec_model, dataset = train_embeddings(True)

index2word = word2vec_model.wv.index2word
vocab = word2vec_model.wv.vocab

print(dataset.shape)