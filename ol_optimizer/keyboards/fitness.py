import numpy as np

from ol_optimizer.keyboards.distances import get_dists_vec
from ol_optimizer.config import A_S_DEFAULT, A_H_DEFAULT, B_H_DEFAULT


def weighted_average_obj_func(bigram_probs_vec: np.array, dists_vec: np.array) -> float:
    return bigram_probs_vec @ dists_vec


def weighted_average_fitness_func(individual: list, bigram_probs: list, bigram_probs_vec: np.array, dist_func='square',
                                  a_s=A_S_DEFAULT, a_h=A_H_DEFAULT, b_h=B_H_DEFAULT):
    dists_vec = get_dists_vec(bigram_probs, individual, dist_func=dist_func, a_s=a_s, a_h=a_h, b_h=b_h)
    return weighted_average_obj_func(bigram_probs_vec, dists_vec),
