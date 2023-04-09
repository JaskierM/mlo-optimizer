from mlo_optimizer.config import A_H_DEFAULT, A_S_DEFAULT, B_H_DEFAULT

import numpy as np


def square_dist(x1: tuple, x2: tuple, a_s: float = A_S_DEFAULT) -> float:
    return a_s * np.linalg.norm(np.array(x2) - np.array(x1))


def hex_dist(x1: tuple, x2: tuple, a_h: float = A_H_DEFAULT, b_h: float = B_H_DEFAULT) -> float:
    h_dist = 2 * (x1[1] - x2[1]) - (x1[0] % 2 - x2[0] % 2)
    v_dist = x1[0] - x2[0]
    return ((a_h * h_dist)**2 + (b_h * v_dist)**2) ** (1/2)


def key_index(key: str, keyboard_matrix: list) -> np.array:
    indexes = []

    for i in range(len(keyboard_matrix)):
        for i_1 in range(len(keyboard_matrix[i])):
            if type(keyboard_matrix[i][i_1]) != list:
                if keyboard_matrix[i][i_1] == key:
                    indexes.append((i, i_1))
            else:
                for key_j in keyboard_matrix[i][i_1]:
                    if key_j == key:
                        indexes.append((i, i_1))

    return np.array(indexes)


def get_dists_vec(bigram_probs: list, keyboard_matrix: list, dist_func: str = 'square', a_s: float = A_S_DEFAULT,
                  a_h: float = A_H_DEFAULT, b_h: float = B_H_DEFAULT) -> np.array:
    result_vec = []

    for bigram in bigram_probs:
        key_indexes_1 = key_index(bigram[0][0], keyboard_matrix)
        key_indexes_2 = key_index(bigram[0][1], keyboard_matrix)

        dists = []
        for key_index_1 in key_indexes_1:
            for key_index_2 in key_indexes_2:
                if dist_func == 'square':
                    dists.append(square_dist(key_index_1, key_index_2, a_s=a_s))
                elif dist_func == 'hex':
                    dists.append(hex_dist(key_index_1, key_index_2, a_h=a_h, b_h=b_h))

        try:
            result_vec.append(min(dists))
        except ValueError:
            print(f'ValueError: At least one element of the counted elements: \'{bigram[0][0]}\' or \'{bigram[0][1]}\' '
                  f'does not match the element in the keyboard')
            exit(1)

    return np.array(result_vec)