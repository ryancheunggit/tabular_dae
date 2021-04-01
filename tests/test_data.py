import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tabular_dae.data import FreqLabelEncoder, DataFrameParser, SingleDataset


def test_freqlabelencoder():
    a = ['1', '1', '2', '3', '4', '4', '5', '5', '5']
    b = FreqLabelEncoder().fit_transform(a)
    assert all(b == np.array([1, 1, 0, 0, 1, 1, 2, 2, 2]))


def test_dataframeparser():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data', 'titanic.csv'))
    parser = DataFrameParser().fit(df)

    # full dataframe
    M = parser.transform(df)
    print(df[parser._column_order].head(2))
    print(M[:2, :])
    assert M.shape == df.shape

    # pandas series
    x = df.iloc[0, :]
    a = parser.transform_single(x)
    print(a)
    assert all(a == M[0, :])

    # dictionary
    x = df.iloc[~0, :].to_dict()
    a = parser.transform_single(x)
    print(a)
    assert all(a == M[~0, :])


def test_dataloader():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data', 'titanic.csv'))
    parser = DataFrameParser().fit(df)
    data = parser.transform(df)
    ds = SingleDataset(data, parser.datatype_info())

    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size)
    X = next(iter(dl))
    assert list(X.keys()) == ['bins', 'cats', 'nums']
    assert X['bins'].shape == torch.Size([batch_size, parser.n_bins])
    assert X['cats'].shape == torch.Size([batch_size, parser.n_cats])
    assert X['nums'].shape == torch.Size([batch_size, parser.n_nums])
