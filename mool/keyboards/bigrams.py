import numpy as np
import pandas as pd
import nltk

from tqdm import tqdm
from config import BATCH_SIZE
from typing import Tuple


def tokenize_by_letters(texts: pd.Series) -> list:
    tokenized_text = []

    print('Tokenization...')
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        texts_batch = texts[i:i + BATCH_SIZE]

        for sentence in texts_batch:
            for letter in str(sentence).lower():
                tokenized_text.append(letter)

    return tokenized_text


def get_bigram_probs(tokenized_text: list) -> list:
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


def filter_bigram_probs(bigram_probs: list, counted_keys: list) -> list:
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
        if new_bigram[0] in counted_keys and new_bigram[1] in counted_keys:
            filtered_bigram_probs.append(((new_bigram[0], new_bigram[1]), bigram[1]))

    return filtered_bigram_probs


def get_bigram_probs_vec(bigram_probs: list) -> np.array:
    if bigram_probs is not None:
        return np.array([elem[1] for elem in bigram_probs])


def get_bigram_probs_with_vec(texts: pd.Series, counted_elems: list) -> Tuple[list, np.array]:
    tokenized_text = tokenize_by_letters(texts)
    bigram_probs = get_bigram_probs(tokenized_text)
    filtered_bigram_probs = filter_bigram_probs(bigram_probs, counted_elems)
    filtered_bigram_probs_vec = get_bigram_probs_vec(filtered_bigram_probs)

    return filtered_bigram_probs, filtered_bigram_probs_vec
