from data.read import read_dir
from keyboards.bigrams import get_bigram_probs_with_vec

from config import RU_LANG_PART_DIR, EN_LANG_PART_DIR, RU_COUNTED_ELEMS, EN_COUNTED_ELEMS


def main():
    texts = read_dir(EN_LANG_PART_DIR)
    print(len(get_bigram_probs_with_vec(texts, EN_COUNTED_ELEMS)[0]))


if __name__ == '__main__':
    main()
