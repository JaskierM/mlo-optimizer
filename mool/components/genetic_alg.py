import numpy as np
import random

from copy import deepcopy
from deap import creator


def random_matrix(init_matrix: list, permutable_elems: list):
    permutable_elems_copy = deepcopy(permutable_elems)
    random.shuffle(permutable_elems_copy)

    np_array = np.array(init_matrix)
    dims = np_array.shape
    vec = np_array.flatten()

    for i in range(len(vec)):
        if vec[i] is None:
            try:
                random_elem = permutable_elems_copy.pop()
                vec[i] = random_elem
            except IndexError:
                break

    res_matrix = vec.reshape(dims).tolist()
    return creator.Individual(res_matrix)


def mate_matrix(individual_1: list, individual_2: list, permutable_elems: list):
    np_array_1, np_array_2 = np.array(individual_1), np.array(individual_2)
    dims_1, dims_2 = np_array_1.shape, np_array_2.shape

    vec_1 = np_array_1.flatten().tolist()
    vec_2 = np_array_2.flatten().tolist()

    size = len(vec_1)

    assert size == len(vec_2), 'Mismatched dimensions of matrices when trying to crossbreeding'

    vec_1_indexes, vec_2_indexes = {}, {}
    for i in range(size):
        vec_1_indexes[i] = vec_1[i]
        vec_2_indexes[i] = vec_2[i]

    vec_1 = list(vec_1_indexes.keys())
    vec_2 = list(vec_2_indexes.keys())

    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes_1, holes_2 = [True if key in permutable_elems else False for key in vec_1], \
        [True if key in permutable_elems else False for key in vec_2]

    for i in range(size):
        if i < a or i > b:
            holes_1[vec_2[i]] = False
            holes_2[vec_1[i]] = False

    temp_1, temp_2 = vec_1, vec_2
    k_1, k_2 = b + 1, b + 1
    for i in range(size):
        if not holes_1[temp_1[(i + b + 1) % size]]:
            vec_1[k_1 % size] = temp_1[(i + b + 1) % size]
            k_1 += 1

        if not holes_2[temp_2[(i + b + 1) % size]]:
            vec_2[k_2 % size] = temp_2[(i + b + 1) % size]
            k_2 += 1

    for i in range(a, b + 1):
        vec_1[i], vec_2[i] = vec_2[i], vec_1[i]

    res_matrix_1 = np.array([vec_1_indexes[i] for i in vec_1]).reshape(dims_1).tolist()
    res_matrix_2 = np.array([vec_2_indexes[i] for i in vec_2]).reshape(dims_2).tolist()

    return creator.Individual(res_matrix_1), creator.Individual(res_matrix_2)


def mutate_matrix(individual: list, permutable_elems: list):
    np_array = np.array(individual)
    dims = np_array.shape

    vec = np.array(individual).flatten()

    size = len(vec)
    indpb = 1.0 / size

    for i in range(size):
        if vec[i] in permutable_elems and np.random.random() < indpb:
            while True:
                swap_i = random.randint(0, size - 2)

                if swap_i >= i:
                    swap_i += 1
                if vec[swap_i] not in permutable_elems:
                    continue
                vec[i], vec[swap_i] = vec[swap_i], vec[i]
                break

    res_matrix = vec.reshape(dims).tolist()

    return creator.Individual(res_matrix),
