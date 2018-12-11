import numpy as np
from data_loader import DataLoader

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize


# =======================
# Calculate perplexity
# =======================

def make_corpus(torchtext_data):
    corpus = [' '.join(data.text).replace("<br />", "").replace("< br />", "") for data in torchtext_data]
    return corpus


def get_count_dictionary(data_loader=None):
    if not data_loader:
        data_loader = DataLoader()
        data_loader.load_data()

    print('building vectorizer...')
    train, valid = data_loader.large_train_valid()
    train_corpus = make_corpus(train)

    uni_vectorizer = CountVectorizer(max_features=20000)
    bi_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=20000)

    _ = uni_vectorizer.fit(train_corpus)
    _ = bi_vectorizer.fit(train_corpus)

    return uni_vectorizer.vocabulary_, bi_vectorizer.vocabulary_


def get_probability(target, context, unigram_dict, bigram_dict, alpha, vocab_size):
    """
    :param target: The word whose probability of seeing is being computed given the context
    :param context: The word directly preceeding the target word
    :param unigram_dict: A dictionary containing unigrams as keys and their respective counts as values
    :param bigram_dict: A dictionary containing bigrams as keys and their respective counts as values
    :param alpha: The amount of additive smoothing being applied
    :param vocab_size: The size of the training vocabulary
    :return: The probability of seeing the target word given the context
    """
    bigram = context + ' ' + target
    bigram_count = bigram_dict[bigram] if bigram in bigram_dict else 0
    unigram_count = unigram_dict[context] if context in unigram_dict else 0
    return (bigram_count + alpha) / (unigram_count + alpha * vocab_size)


def calculate_perplexity(text, unigram_dict=None, bigram_dict=None, alpha=0.1, vocab_size=20000):
    """Calculate perplexity for the given text.

    :param text: Text to be evaluated
    :param unigram_dict: unigram vocabulary dictionary
    :param bigram_dict: bigram vocabulary dictionary
    :param alpha: The amount of additive smoothing being applied
    :param vocab_size: Total vocabulary size
    """
    if not (unigram_dict and bigram_dict):
        unigram_dict, bigram_dict = get_count_dictionary()
    tokens = text.split(" ")
    probs = [np.log(get_probability(tokens[i], tokens[i-1], unigram_dict, bigram_dict, alpha, vocab_size)) for i in range(1, len(tokens))]
    N = len(tokens) - 1

    return np.exp(np.sum(probs) / -N)


# =========================
# Part of Speech Tagging
# =========================

def check_pos(idx1, idx2, data_loader):
    text1 = word_tokenize(data_loader.TEXT.vocab.itos[idx1])
    text2 = word_tokenize(data_loader.TEXT.vocab.itos[idx2])
    if len(text1) > 0 and len(text2) > 0:
        tag1 = nltk.pos_tag(text1)[0][1]
        tag2 = nltk.pos_tag(text2)[0][1]
        return tag1 == tag2
    else:
        return False