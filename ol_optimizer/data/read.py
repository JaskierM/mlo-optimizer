import pathlib
import pandas as pd


def read_dir(directory: str, pattern: str = '*.txt'):
    path = pathlib.Path(directory)
    assert path.exists(), f'Directory "{directory}" does not exist'

    texts = pd.Series()

    for file in path.glob(pattern):
        print(f'Processing {file}')

        cur_series = pd.read_csv(file, sep='\r\n', header=None, engine='python')[0]
        texts = pd.concat((texts, cur_series))

    return texts.reset_index(drop=True)
