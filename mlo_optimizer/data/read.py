import pathlib

import pandas as pd


def read_dir(directory: str, pattern: str = '*.txt'):
    """Reads all lines from text files

    :param directory: Path to directory with text files
    :type directory: str
    :param pattern: File format to read
    :type pattern: str
    :return: Pandas series with read lines
    """
    path = pathlib.Path(directory)
    assert path.exists(), f'Directory "{directory}" does not exist'

    texts = pd.Series()

    for text_file in path.glob(pattern):
        print(f'Processing {text_file}')

        cur_series = pd.read_csv(text_file, sep='\r\n', header=None, engine='python')[0]
        texts = pd.concat((texts, cur_series))

    return texts.reset_index(drop=True)
