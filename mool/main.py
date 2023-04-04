from data.read import read_dir
from keyboards.bigrams import get_bigram_probs_with_vec
from keyboards.distances import get_dists_vec
from keyboards.fitness import weighted_average_fitness

from config import EN_LANG_PART_DIR, EN_COUNTED_ELEMS, EN_TEST_HEX_KEYBOARD


def main():
    en_texts = read_dir(EN_LANG_PART_DIR)
    bigram_probs, bigram_probs_vec = get_bigram_probs_with_vec(en_texts, EN_COUNTED_ELEMS)
    dists_vec = get_dists_vec(bigram_probs, EN_TEST_HEX_KEYBOARD)
    print(weighted_average_fitness(bigram_probs_vec, dists_vec))


if __name__ == '__main__':
    main()
