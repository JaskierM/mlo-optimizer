RU_LANG_PART_DIR = '../data/raw/ru/'
EN_LANG_PART_DIR = '../data/raw/en/'

BATCH_SIZE = 128

RU_COUNTED_ELEMS = [chr(i) for i in range(1072, 1104)] + ['ё'] + ['space', 'enter'] + ['.', ',']
RU_PERMUTABLE_ELEMS = [chr(i) for i in range(1072, 1104)] + ['ё'] + ['.', ',']

EN_COUNTED_ELEMS = [chr(i) for i in range(97, 123)] + ['space', 'enter'] + ['.', ',', '!', '?', '-', ':', ';', '(', ')']
EN_PERMUTABLE_ELEMS = [chr(i) for i in range(97, 123)] + ['.', ',', '!', '?', '-', ':', ';', '(', ')']
