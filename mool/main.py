from optimizer import Optimizer

from config import INIT_HEX_KEYBOARD, EN_COUNTED_ELEMS, EN_PERMUTABLE_ELEMS, EN_LANG_PART_DIR


def main():
    optimizer = Optimizer(
        INIT_HEX_KEYBOARD,
        EN_COUNTED_ELEMS,
        EN_PERMUTABLE_ELEMS,
        fitness_func='hex',
        max_generation=10,
        population_size=30,
    )

    optimizer.fit_bigrams(EN_LANG_PART_DIR)
    best_matrix = optimizer.optimize()
    print(best_matrix)


if __name__ == '__main__':
    main()
