from data.read import read_dir

from config import RU_LANG_PART_DIR, EN_LANG_PART_DIR


def main():
    print(read_dir(EN_LANG_PART_DIR))
    print(read_dir(RU_LANG_PART_DIR))


if __name__ == '__main__':
    main()
