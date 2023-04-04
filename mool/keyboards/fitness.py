import numpy as np


def weighted_average_fitness(bigram_probs_vec: np.array, dists_vec: np.array) -> float:
    return bigram_probs_vec @ dists_vec
