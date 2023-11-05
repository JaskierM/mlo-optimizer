import pathlib

from deap import base, creator, tools

from mlo_optimizer.components.algelitism import ea_simple_elitism
from mlo_optimizer.components.genetic_alg import mate_matrix, mutate_matrix, random_matrix
from mlo_optimizer.config import A_H_DEFAULT, A_S_DEFAULT, B_H_DEFAULT, FITNESS_FUNC_DEFAULT, \
    HALL_OF_FAME_SIZE_DEFAULT, MAX_GENERATION_DEFAULT, MINIMIZATION_DEFAULT, POPULATION_SIZE_DEFAULT, \
    P_CROSSOVER_DEFAULT, P_MUTATION_DEFAULT, TOURN_SIZE_DEFAULT
from mlo_optimizer.data.read import read_dir
from mlo_optimizer.descriptors.bigram_probs_descriptor import BigramProbsDescriptor
from mlo_optimizer.descriptors.coef_descriptor import CoefficientDescriptor
from mlo_optimizer.descriptors.list_descriptor import ListDescriptor
from mlo_optimizer.descriptors.probability_descriptor import ProbabilityDescriptor
from mlo_optimizer.descriptors.size_descriptor import SizeDescriptor
from mlo_optimizer.keyboards.bigrams import get_bigram_probs_with_vec
from mlo_optimizer.keyboards.fitness import weighted_average_fitness_func

import numpy as np

script_dir = pathlib.Path(__file__).parent.resolve()


class Optimizer:
    """Main class that tunes the components of the genetic algorithm and implements the optimizer

    :param init_matrix: Initial initialization matrix with elements (permutable and not counted)
    :type init_matrix: list
    :param counted_elems: Set of elements taken into account in the objective function
    :type counted_elems: list
    :param permutable_elems: The set of elements that are rearranged during crossover and mutation
    :type permutable_elems: list
    :param fitness_func: Objective function that receives an individual as input and returns a fitness score
    :type fitness_func: callable
    :param fitness_func_kwargs: Named arguments for target function
    :type fitness_func_kwargs: dict
    :param minimization: Minimization and Maximization Flag of the Objective Function
    :type minimization: bool
    :param a_s: Half side of square button (when fitness_func='square')
    :type a_s: float
    :param a_h: Distance from the middle of a hexagonal key to the middle of its side (when fitness_func='hex')
    :type a_h: float
    :param b_h: Distance from the middle of the hexagonal key to the middle of the side of the bottom key (when
    fitness_func='hex')
    :type b_h: float
    :param population_size: Population size in one generation
    :type population_size: int
    :param p_crossover: Crossbreeding probability
    :type p_crossover: float
    :param p_mutation: Mutation probability
    :type p_mutation: float
    :param max_generation: Maximum number of generations
    :type max_generation: int
    :param tourn_size: sample size for tournament selection
    :type tourn_size: int
    :param hall_of_fame_size: Number of best individuals obtained after the completion of the optimization
    :type hall_of_fame_size: int
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
                 fitness_func: callable = FITNESS_FUNC_DEFAULT,
                 fitness_func_kwargs: dict = None,
                 minimization: bool = MINIMIZATION_DEFAULT,
                 a_s: float = A_S_DEFAULT,
                 a_h: float = A_H_DEFAULT,
                 b_h: float = B_H_DEFAULT,
                 population_size: int = POPULATION_SIZE_DEFAULT,
                 max_generation: int = MAX_GENERATION_DEFAULT,
                 p_crossover: float = P_CROSSOVER_DEFAULT,
                 p_mutation: float = P_MUTATION_DEFAULT,
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
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.tourn_size = tourn_size
        self.hall_of_fame_size = hall_of_fame_size

        self.bigram_probs = None
        self.bigram_probs_vec = None

    def fit_bigrams(self, lang_part_dir: str):
        """Reads text files and construct bigram probability vectors from them

        :param lang_part_dir: Folder with text files
        :type lang_part_dir: str
        """
        texts = read_dir(lang_part_dir)
        self.bigram_probs, self.bigram_probs_vec = get_bigram_probs_with_vec(texts, self.counted_elems)

    def optimize(self):
        """Collects all components and runs optimization

        :return: Matrices of the best individuals (quantity depends on the parameter hall_of_fame_size)
        """
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
            toolbox.register('evaluate', self.__fitness_func, bigram_probs=self.bigram_probs,
                             bigram_probs_vec=self.bigram_probs_vec, dist_func='hex', a_h=self.a_h, b_h=self.b_h,
                             **self.__fitness_func_kwargs)

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
        elif isinstance(value, dict):
            raise TypeError('Attribute "fitness_func_kwargs" must be represented as Python dict')
        else:
            self.__fitness_func_kwargs = value

    @property
    def minimization(self):
        return self.__minimization

    @minimization.setter
    def minimization(self, value):
        if isinstance(value, bool):
            raise TypeError('Attribute "minimization" must be represented as boolean')
        self.__minimization = value
