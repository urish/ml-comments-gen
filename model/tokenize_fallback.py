import string
from keras.preprocessing.text import text_to_word_sequence

def fallback_tokenizer_dictionary(prefix = '>'):
    return [' '.join([prefix + l for l in string.ascii_letters + string.digits])]

def tokenize_with_fallback(tokenizer, strings, max_words, prefix = '>'):
    """
    Tokenize the given strings using keras' built-in tokenizer.
    words not in the dictionary will be replaced by individual characters. 
    """
    def tokenize_single(s):
        words = text_to_word_sequence(s, filters=tokenizer.filters, split=tokenizer.split)
        mapped = []
        for word in words:
            if word in tokenizer.word_index and tokenizer.word_index[word] < max_words:
                mapped.append(word)
            else:
                mapped += [prefix + l for l in word]
        return ' '.join(mapped)

    strings = map(tokenize_single, strings)
    return tokenizer.texts_to_sequences(strings)
