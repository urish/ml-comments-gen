import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


def max_length(seqs, max_seq_len=500):
    length = max(len(s) for s in seqs)
    return length if length < max_seq_len else max_seq_len


def prepare_dataset(raw_x1, raw_x2, max_seq_len, num_classes, name):
    x1_maxlen = max_length(raw_x1, max_seq_len)
    x2_maxlen = max_length(raw_x2, max_seq_len)

    print("Preparing {} Dataset".format(name))
    print("[Pepare Dataset] Max Sequence x1:", x1_maxlen)
    print("[Pepare Dataset] Max Sequence x2:", x2_maxlen)

    maxlen = max(x1_maxlen, x2_maxlen)

    print("[Pepare Dataset] Max Sequence:", maxlen)

    x1, x2, y = list(), list(), list()
    for x2_idx, seq in enumerate(raw_x2):

        x1_seq = pad_sequences(
            [raw_x1[x2_idx]], maxlen=maxlen, padding="post", truncating="post"
        )[0]

        for i in range(1, len(seq)):
            # add AST
            x1.append(x1_seq)

            # add the entire sequence to the input and only keep the next word for the output
            in_seq, out_seq = seq[:i], seq[i]

            # apply padding and encode sequence
            in_seq = pad_sequences(
                [in_seq], maxlen=maxlen, padding="post", truncating="post"
            )[0]

            # one hot encode output sequence
            out_seq = to_categorical([out_seq], num_classes=num_classes)[0]
            y.append(out_seq)

            # cut the input seq to fixed length
            x2.append(in_seq)

    x1, x2, y = np.array(x1), np.array(x2), np.array(y)
    return x1, x2, y


def create_embedding_matrix(dim, word_index, word_vectors):
    max_num_words = len(word_index) + 1

    # we initialize the matrix with zeros
    embedding_matrix = np.zeros((max_num_words, dim))

    for word, i in word_index.items():
        if i >= max_num_words:
            continue

        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except:
            pass

    return embedding_matrix


"""
This function can be used to debug the preprocessed dataset and print out
encoded inputs as well as the label.

Be aware of calling this function on a large dataset as it will produce dozens
of print statemenets.
"""


def check_encoding(x1, x2, y, x1_idx2word, x2_idx2word):
    print("--- Check Encoding ---")
    for idx, x1_encoded in enumerate(x1):
        x1_decoded = list()
        x2_decoded = ""
        y_decoded = ""

        # decode x1
        for x1_seq_idx in x1_encoded:
            if x1_seq_idx != 0:
                x1_decoded.append(x1_idx2word[x1_seq_idx])

        # decode x2
        for x2_seq_idx in reversed(x2[idx]):
            if x2_seq_idx != 0:
                x2_decoded = x2_idx2word[x2_seq_idx] + " " + x2_decoded

        y_encoded = y[idx]
        y_argmax = np.argmax(y_encoded)
        if y_argmax != 0:
            y_decoded = x2_idx2word[y_argmax]

        print("X1", idx, " ".join(x1_decoded))
        print("X2", idx, x2_decoded)
        print("Y ", idx, y_decoded)
        print("---")


def save_tokenizer(tokenizer, file_path):
    with open(file_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer successfully saved ({})".format(file_path))


def load_tokenizer(file_path):
    with open(file_path, "rb") as handle:
        tokenizer = pickle.load(handle)
        print("Tokenizer loaded ({})".format(file_path))
        return tokenizer
