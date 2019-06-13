import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

dataframe = pd.read_json('../data/dataset.json', lines=True)

tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts(dataframe['ast'].sum())

print(tokenizer.word_index)
