import numpy as np
import pathlib

from deap import base, creator, tools

from ol_optimizer.data.read import read_dir
from ol_optimizer.keyboards.bigrams import get_bigram_probs_with_vec
from ol_optimizer.keyboards.fitness import weighted_average_fitness_func
from ol_optimizer.components.genetic_alg import random_matrix, mate_matrix, mutate_matrix
from ol_optimizer.components.algelitism import ea_simple_elitism
from ol_optimizer.config import FITNESS_FUNC_DEFAULT, MINIMIZATION_DEFAULT, A_S_DEFAULT, A_H_DEFAULT, B_H_DEFAULT, \
    POPULATION_SIZE_DEFAULT, P_CROSSOVER_DEFAULT, P_MUTATION_DEFAULT, MAX_GENERATION_DEFAULT, TOURN_SIZE_DEFAULT, \
    HALL_OF_FAME_SIZE_DEFAULT

script_dir = pathlib.Path(__file__).parent.resolve()


class Optimizer:
    def __init__(self,
                 init_matrix: list,
                 counted_elems: list,
                 permutable_elems: list,
                 fitness_func=FITNESS_FUNC_DEFAULT,
                 fitness_func_kwargs: dict = None,
                 minimization: bool = MINIMIZATION_DEFAULT,
                 a_s: float = A_S_DEFAULT,
                 a_h: float = A_H_DEFAULT,
                 b_h: float = B_H_DEFAULT,
                 population_size: int = POPULATION_SIZE_DEFAULT,
                 p_crossover: float = P_CROSSOVER_DEFAULT,
                 p_mutation: float = P_MUTATION_DEFAULT,
                 max_generation: int = MAX_GENERATION_DEFAULT,
                 tourn_size: int = TOURN_SIZE_DEFAULT,
                 hall_of_fame_size: int = HALL_OF_FAME_SIZE_DEFAULT):

        self.__init_matrix = init_matrix
        self.__counted_elems = counted_elems
        self.__permutable_elems = permutable_elems
        self.__fitness_func = fitness_func
        self.__fitness_func_kwargs = fitness_func_kwargs
        self.__minimization = minimization
        self.__a_s = a_s
        self.__a_h = a_h
        self.__b_h = b_h
        self.__population_size = population_size
        self.__p_crossover = p_crossover
        self.__p_mutation = p_mutation
        self.__max_generation = max_generation
        self.__tourn_size = tourn_size
        self.__hall_of_fame_size = hall_of_fame_size

        self.__bigram_probs = None
        self.__bigram_probs_vec = None

        assert self.__fitness_func in ('square', 'hex') or callable(self.__fitness_func), \
            'The distance function must be "square", "hex" or your own function'

        if self.__fitness_func_kwargs is None:
            self.__fitness_func_kwargs = {}

    def fit_bigrams(self, lang_part_dir: str):
        texts = read_dir(lang_part_dir)
        self.__bigram_probs, self.__bigram_probs_vec = get_bigram_probs_with_vec(texts, self.__counted_elems)

    def optimize(self):
        if self.__minimization:
            weight = -1.0
        else:
            weight = 1.0

        creator.create('FitnessMin', base.Fitness, weights=(weight,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('randomMatrix', random_matrix, self.__init_matrix, self.__permutable_elems)
        toolbox.register('populationCreator', tools.initRepeat, list, toolbox.randomMatrix)

        population = toolbox.populationCreator(n=self.__population_size)

        if self.__fitness_func == 'square':
            assert self.__bigram_probs is not None and self.__bigram_probs_vec is not None, \
                'Before optimization using the "square" function, it is necessary to determine the weights of the ' \
                'bigram using the "fit_bigrams" function'
            toolbox.register('evaluate', weighted_average_fitness_func, bigram_probs=self.__bigram_probs,
                             bigram_probs_vec=self.__bigram_probs_vec, dist_func='square', a_s=self.__a_s)
        elif self.__fitness_func == 'hex':
            assert self.__bigram_probs is not None and self.__bigram_probs_vec is not None, \
                'Before optimization using the "hex" function, it is necessary to determine the weights of the ' \
                'bigram using the "fit_bigrams" function'
            toolbox.register('evaluate', weighted_average_fitness_func, bigram_probs=self.__bigram_probs,
                             bigram_probs_vec=self.__bigram_probs_vec, dist_func='hex', a_h=self.__a_h, b_h=self.__b_h)
        else:
            toolbox.register('evaluate', self.__fitness_func, **self.__fitness_func_kwargs)

        toolbox.register('select', tools.selTournament, tournsize=self.__tourn_size)
        toolbox.register('mate', mate_matrix, permutable_elems=self.__permutable_elems)
        toolbox.register('mutate', mutate_matrix, permutable_elems=self.__permutable_elems)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)

        hof = tools.HallOfFame(self.__hall_of_fame_size)

        ea_simple_elitism(
            population,
            toolbox,
            cxpb=self.__p_crossover,
            mutpb=self.__p_mutation,
            ngen=self.__max_generation,
            halloffame=hof,
            stats=stats,
            verbose=True)

        best_matrices = hof.items[0]

        return best_matrices
