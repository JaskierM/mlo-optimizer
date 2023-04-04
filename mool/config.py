BATCH_SIZE = 128

A_S_DEFAULT = 1
A_H_DEFAULT = 0.537634
B_H_DEFAULT = 0.930605

FITNESS_FUNC_DEFAULT = 'square'
MINIMIZATION_DEFAULT = True
POPULATION_SIZE_DEFAULT = 50
MAX_GENERATION_DEFAULT = 50
P_CROSSOVER_DEFAULT = 0.9
P_MUTATION_DEFAULT = 0.2
TOURN_SIZE_DEFAULT = 3
HALL_OF_FAME_SIZE_DEFAULT = 1

# Debug
RU_LANG_PART_DIR = '../data/raw/ru/'
EN_LANG_PART_DIR = '../data/raw/en/'

RU_COUNTED_ELEMS = [chr(i) for i in range(1072, 1104)] + ['ё'] + ['space', 'enter'] + ['.', ',']
RU_PERMUTABLE_ELEMS = [chr(i) for i in range(1072, 1104)] + ['ё'] + ['.', ',']

EN_COUNTED_ELEMS = [chr(i) for i in range(97, 123)] + ['space', 'enter'] + ['.', ',', '!', '?', '-', ':', ';', '(', ')']
EN_PERMUTABLE_ELEMS = [chr(i) for i in range(97, 123)] + ['.', ',', '!', '?', '-', ':', ';', '(', ')']

INIT_HEX_KEYBOARD = [
    ['inv', 'lang', None, None, None, None, '?123'],
    ['settings', None, None, None, None, None, 'backspace'],
    ['inv', None, None, None, None, None, None],
    [None, None, None, 'space', None, None, 'enter'],
    ['inv', None, None, None, None, None, None],
    ['move', None, None, None, None, None, 'capslock'],
    ['inv', 'exit', None, None, None, None, 'shift'],
]

RU_TEST_HEX_KEYBOARD = [
    ['inv', 'lang', 'а', 'б', 'в', 'г', '?123'],
    ['settings', 'д', 'е', 'ё', 'ж', 'з', 'backspace'],
    ['inv', 'и', 'й', 'к', 'л', 'м', 'н'],
    ['о', 'п', 'р', 'space', 'с', 'т', 'enter'],
    ['inv', 'у', 'ф', 'х', 'ц', 'ч', 'ш'],
    ['move', 'щ', 'ъ', 'ы', 'ь', 'э', 'capslock'],
    ['inv', 'exit', 'ю', 'я', '.', ',', 'shift'],
]

EN_TEST_HEX_KEYBOARD = [
    ['inv', 'lang', 'a', 'b', 'c', 'd', '?123'],
    ['settings', 'e', 'f', 'g', 'h', 'i', 'backspace'],
    ['inv', 'j', 'k', 'l', 'm', 'n', 'o'],
    ['p', 'q', 'r', 'space', 's', 't', 'enter'],
    ['inv', 'u', 'v', 'w', 'x', 'y', 'z'],
    ['move', '.', ',', '!', '?', ':', 'capslock'],
    ['inv', 'exit', ';', '-', '(', ')', 'shift'],
]
