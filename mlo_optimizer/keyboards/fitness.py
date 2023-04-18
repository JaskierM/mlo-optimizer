from mlo_optimizer.config import A_H_DEFAULT, A_S_DEFAULT, B_H_DEFAULT
from mlo_optimizer.keyboards.distances import get_dists_vec

import numpy as np


def weighted_average_obj_func(bigram_probs_vec: np.array, dists_vec: np.array) -> float:
    """Calculates fitness using a vector of bigrams and distances

    :param bigram_probs_vec: Bigram probability vector
    :type bigram_probs_vec: class:`numpy.array`
    :param dists_vec: Distance vector between each bigram from bigram probability vector
    :type dists_vec: class:`numpy.array`
    :return: Fitness assessment
    """
    return bigram_probs_vec @ dists_vec


def weighted_average_fitness_func(individual: list, bigram_probs: list, bigram_probs_vec: np.array, dist_func='square',
                                  a_s=A_S_DEFAULT, a_h=A_H_DEFAULT, b_h=B_H_DEFAULT):
    """Calculates the fitness score for the current individual

    :param individual: Individual
    :type individual: list
    :param bigram_probs: List of bigrams of the form: ((first symbol, next symbol), probability)
    :type bigram_probs: list
    :param bigram_probs_vec: Bigram probability vector
    :type bigram_probs_vec: class:`numpy.array`
    :param dist_func: Function to calculate the distance between two keys ('square' or 'hex')
    :type dist_func: str
    :param a_s: Half side of square button (when fitness_func='square')
    :type a_s: float
    :param a_h: Distance from the middle of a hexagonal key to the middle of its side (when fitness_func='hex')
    :type a_h: float
    :param b_h: Distance from the middle of the hexagonal key to the middle of the side of the bottom key (when
    fitness_func='hex')
    :type b_h: float
    :return: Fitness assessment
    """
    dists_vec = get_dists_vec(bigram_probs, individual, dist_func=dist_func, a_s=a_s, a_h=a_h, b_h=b_h)
    return (weighted_average_obj_func(bigram_probs_vec, dists_vec),)
