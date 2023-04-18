from typing import Tuple

from mlo_optimizer.config import BATCH_SIZE

import nltk

import numpy as np

import pandas as pd

from tqdm import tqdm


def tokenize_by_letters(texts: pd.Series) -> list:
    """Separates texts by character

    :param texts: Series with strings from text files
    :type texts: class:`pandas.Series`
    :return: List of all characters
    """
    tokenized_text = []

    print('Tokenization...')
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        texts_batch = texts[i:i + BATCH_SIZE]

        for sentence in texts_batch:
            for letter in str(sentence).lower():
                tokenized_text.append(letter)

    return tokenized_text


def get_bigram_probs(tokenized_text: list) -> list:
    """Calculates the probabilities of bigrams from each combination of two keys

    :param tokenized_text: List of all texts separated by characters
    :type tokenized_text: list
    :return: List of bigrams of the form: ((first symbol, next symbol), probability)
    """
    print('Getting bigrams...')
    bigrams = nltk.bigrams(tokenized_text)
    print('Calculation of bigram frequencies...')
    freq_dists = nltk.FreqDist(bigrams)
    print('Calculation of bigram probabilities...')
    prob_dists = nltk.MLEProbDist(freq_dists)

    bigram_probs = []
    print('Combining bigrams and their frequencies')
    for sample in tqdm(freq_dists.keys()):
        bigram_probs.append((sample, prob_dists.prob(sample)))

    return bigram_probs


def filter_bigram_probs(bigram_probs: list, counted_elems: list) -> list:
    """Removes bigrams that are not in the counted_keys

    :param bigram_probs: List of bigrams of the form: ((first symbol, next symbol), probability)
    :type bigram_probs list
    :param counted_elems: Set of elements taken into account in the objective function
    :type counted_elems: list
    :return: List of filtered bigrams
    """
    filtered_bigram_probs = []

    print('Bigram filtering...')
    for bigram in tqdm(bigram_probs):
        new_bigram = [bigram[0][0], bigram[0][1]]

        if bigram[0][0] == ' ':
            new_bigram[0] = 'space'
        if bigram[0][1] == ' ':
            new_bigram[1] = 'space'
        if bigram[0][0] == '\n':
            new_bigram[0] = 'enter'
        if bigram[0][1] == '\n':
            new_bigram[1] = 'enter'
        if new_bigram[0] in counted_elems and new_bigram[1] in counted_elems:
            filtered_bigram_probs.append(((new_bigram[0], new_bigram[1]), bigram[1]))

    return filtered_bigram_probs


def get_bigram_probs_vec(bigram_probs: list) -> np.array:
    """Separates the vector of bigrams from the list of bigrams

    :param bigram_probs: List of bigrams of the form: ((first symbol, next symbol), probability)
    :type bigram_probs: list
    :return: Bigram vector
    """
    if bigram_probs is not None:
        return np.array([elem[1] for elem in bigram_probs])


def get_bigram_probs_with_vec(texts: pd.Series, counted_elems: list) -> Tuple[list, np.array]:
    """Launches a full pipeline with the calculation of lists of probabilities of bigrams

    :param texts: Separates texts by character
    :type texts: class:`pandas.Series`
    :param counted_elems: Set of elements taken into account in the objective function
    :type counted_elems: list
    :return: List of filtered bigrams and bigram vector
    """
    tokenized_text = tokenize_by_letters(texts)
    bigram_probs = get_bigram_probs(tokenized_text)
    filtered_bigram_probs = filter_bigram_probs(bigram_probs, counted_elems)
    filtered_bigram_probs_vec = get_bigram_probs_vec(filtered_bigram_probs)

    return filtered_bigram_probs, filtered_bigram_probs_vec
