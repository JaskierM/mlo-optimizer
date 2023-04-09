import pathlib

from deap import base, creator, tools

from mlo_optimizer.components.algelitism import ea_simple_elitism
from mlo_optimizer.components.genetic_alg import mate_matrix, mutate_matrix, random_matrix
from mlo_optimizer.config import A_H_DEFAULT, A_S_DEFAULT, B_H_DEFAULT, FITNESS_FUNC_DEFAULT, \
    HALL_OF_FAME_SIZE_DEFAULT, MAX_GENERATION_DEFAULT, MINIMIZATION_DEFAULT, POPULATION_SIZE_DEFAULT, \
    P_CROSSOVER_DEFAULT, P_MUTATION_DEFAULT, TOURN_SIZE_DEFAULT
from mlo_optimizer.data.read import read_dir
from mlo_optimizer.keyboards.bigrams import get_bigram_probs_with_vec
from mlo_optimizer.keyboards.fitness import weighted_average_fitness_func

import numpy as np

script_dir = pathlib.Path(__file__).parent.resolve()


class ListDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_list(value)
        setattr(instance, self.name, value)

    def verify_list(self, value):
        if type(value) != list:
            raise TypeError(f'Attribute "{self.name[2:]}" must be represented by a Python list')
        if not value:
            raise TypeError(f'Attribute "{self.name[2:]}" must not be empty')


class CoefficientDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_coefficient(value)
        setattr(instance, self.name, value)

    def verify_coefficient(self, value):
        if type(value) not in (int, float):
            raise TypeError(f'Valid types for attribute "{self.name[2:]}" are int and float')
        if value <= 0:
            raise TypeError(f'Attribute "{self.name[2:]}" must be greater than 0')


class SizeDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_size(value)
        setattr(instance, self.name, value)

    def verify_size(self, value):
        if type(value) != int:
            raise TypeError(f'Valid type for attribute "{self.name[2:]}" is int')
        if value <= 0:
            raise TypeError(f'Attribute "{self.name[2:]}" must be greater than 0')


class ProbabilityDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_probability(value)
        setattr(instance, self.name, value)

    def verify_probability(self, value):
        if type(value) not in (int, float):
            raise TypeError(f'Valid types for attribute "{self.name[2:]}" are int and float')
        if value < 0 or value > 1:
            raise TypeError(f'Probability "{self.name[2:]}" must be between 0 and 1')


class BigramProbsDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__x'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)


class Optimizer:
    """Тестовая документация

    :param init_matrix: Класс состояний, хранящий изменяемые флаги и выбранные раскладки
    :type init_matrix: list
    """
    IMPLEMENTED_FITNESS_FUNCS = ('square', 'hex')

    init_matrix = ListDescriptor()
    counted_elems = ListDescriptor()
    permutable_elems = ListDescriptor()
    a_s = CoefficientDescriptor()
    a_h = CoefficientDescriptor()
    b_h = CoefficientDescriptor()
    population_size = SizeDescriptor()
    max_generation = SizeDescriptor()
    tourn_size = SizeDescriptor()
    hall_of_fame_size = SizeDescriptor()
    p_crossover = ProbabilityDescriptor()
    p_mutation = ProbabilityDescriptor()

    bigram_probs = BigramProbsDescriptor()
    get_bigram_probs_with_vec = BigramProbsDescriptor()

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

        self.init_matrix = init_matrix
        self.counted_elems = counted_elems
        self.permutable_elems = permutable_elems
        self.fitness_func = fitness_func
        self.fitness_func_kwargs = fitness_func_kwargs
        self.minimization = minimization
        self.a_s = a_s
        self.a_h = a_h
        self.b_h = b_h
        self.population_size = population_size
        self.max_generation = max_generation
        self.tourn_size = tourn_size
        self.hall_of_fame_size = hall_of_fame_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation

        self.bigram_probs = None
        self.bigram_probs_vec = None

    def fit_bigrams(self, lang_part_dir: str):
        texts = read_dir(lang_part_dir)
        self.bigram_probs, self.bigram_probs_vec = get_bigram_probs_with_vec(texts, self.counted_elems)

    def optimize(self):
        if self.__minimization:
            weight = -1.0
        else:
            weight = 1.0

        creator.create('FitnessMin', base.Fitness, weights=(weight,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('randomMatrix', random_matrix, self.init_matrix, self.permutable_elems)
        toolbox.register('populationCreator', tools.initRepeat, list, toolbox.randomMatrix)

        population = toolbox.populationCreator(n=self.population_size)

        if self.fitness_func == Optimizer.IMPLEMENTED_FITNESS_FUNCS[0]:
            assert self.bigram_probs is not None and self.bigram_probs_vec is not None, \
                'Before optimization using the "square" function, it is necessary to determine the weights of the ' \
                'bigram using the "fit_bigrams" function'
            toolbox.register('evaluate', weighted_average_fitness_func, bigram_probs=self.bigram_probs,
                             bigram_probs_vec=self.bigram_probs_vec, dist_func='square', a_s=self.a_s)
        elif self.__fitness_func == Optimizer.IMPLEMENTED_FITNESS_FUNCS[1]:
            assert self.bigram_probs is not None and self.bigram_probs_vec is not None, \
                'Before optimization using the "hex" function, it is necessary to determine the weights of the ' \
                'bigram using the "fit_bigrams" function'
            toolbox.register('evaluate', weighted_average_fitness_func, bigram_probs=self.bigram_probs,
                             bigram_probs_vec=self.bigram_probs_vec, dist_func='hex', a_h=self.a_h, b_h=self.b_h)
        else:
            toolbox.register('evaluate', self.__fitness_func, **self.__fitness_func_kwargs)

        toolbox.register('select', tools.selTournament, tournsize=self.tourn_size)
        toolbox.register('mate', mate_matrix, permutable_elems=self.permutable_elems)
        toolbox.register('mutate', mutate_matrix, permutable_elems=self.permutable_elems)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)

        hof = tools.HallOfFame(self.hall_of_fame_size)

        ea_simple_elitism(
            population,
            toolbox,
            cxpb=self.p_crossover,
            mutpb=self.p_mutation,
            ngen=self.max_generation,
            halloffame=hof,
            stats=stats,
            verbose=True)

        best_matrices = hof.items[0]

        return best_matrices

    @property
    def fitness_func(self):
        return self.__fitness_func

    @fitness_func.setter
    def fitness_func(self, value):
        if value not in Optimizer.IMPLEMENTED_FITNESS_FUNCS and not callable(value):
            raise TypeError(f'The distance function must be {", ".join(Optimizer.IMPLEMENTED_FITNESS_FUNCS)} or your '
                            f'own function')
        self.__fitness_func = value

    @property
    def fitness_func_kwargs(self):
        return self.__fitness_func_kwargs

    @fitness_func_kwargs.setter
    def fitness_func_kwargs(self, value):
        if value is None:
            self.__fitness_func_kwargs = {}
        elif type(value) != dict:
            raise TypeError('Attribute "fitness_func_kwargs" must be represented as Python dict')
        else:
            self.__fitness_func_kwargs = value

    @property
    def minimization(self):
        return self.__minimization

    @minimization.setter
    def minimization(self, value):
        if type(value) != bool:
            raise TypeError('Attribute "minimization" must be represented as boolean')
        self.__minimization = value
