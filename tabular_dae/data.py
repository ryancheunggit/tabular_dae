import numpy as np
from collections import Counter
from torch.utils.data import Dataset


__all__ = ['StandardScaler', 'LabelEncoder', 'FreqLabelEncoder', 'DataFrameParser', 'SingleDataset']


class StandardScaler(object):
    def __init__(self):
        self.loc = None
        self.scale = None

    def fit(self, x):
        self.loc, self.scale = np.mean(x), np.std(x)
        return self

    def transform(self, x):
        standardized = (np.array(x) - self.loc) / self.scale
        imputed = np.nan_to_num(standardized)
        return imputed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class LabelEncoder(object):
    def __init__(self):
        self.mapping = dict()

    def __len__(self):
        return len(self.mapping)

    def fit(self, x):
        self.mapping = {v: i for i, v in enumerate(set(x))}
        return self

    def transform(self, x):
        return np.array(list(map(self.mapping.__getitem__, x)))


class FreqLabelEncoder(object):
    ''' A composition of label encoding and frequency encoding. Not reversible. '''
    def __init__(self):
        self.freq_counts = None

    def __len__(self):
        return len(self.lbl_encoder)

    def fit(self, x):
        self.freq_counts = Counter(x)
        self.lbl_encoder = LabelEncoder().fit(self.freq_counts.values())
        return self

    def transform(self, x):
        freq_encoded = np.array(list(map(self.freq_counts.__getitem__, x)))
        lbl_encoded = self.lbl_encoder.transform(freq_encoded)
        return lbl_encoded

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class DataFrameParser(object):
    """ Transform dataframe to numpy array for modeling. Not a reversible process.

        It will reshuffle the columns according to datatype: binary->categorical->numerical

        Encoding:
            + Binary variables will be coded as 0, 1
            + Small categorical variables will be label encoded as integers.
            + Categorical variables with large cardinalities will go through count/frequency encoding before label encoding.
            + Numerical will be standardized.

        NaN handling:
            + Fill with mean for numerical. # TODO: Need handling of NaN in categorical? If present in training data is fine.

    """
    def __init__(self, max_cardinality=128):
        self.max_cardinality = max_cardinality
        self.binary_columns = list()
        self.categorical_columns = dict() # variable name to mode mapping
        self._cards = list()
        self.numerical_columns = list()
        self.need_freq_encoding = set()

    def fit(self, dataframe):
        self._original_order = dataframe.columns.tolist()
        self._original_column_to_dtype = column_to_dtype = dataframe.dtypes.to_dict()

        # sort through columns in dataframe.
        for column, datatype in column_to_dtype.items():
            if datatype in ['O', '<U32']:
                cardinality = dataframe[column].nunique(dropna=False)
                if cardinality == 2:
                    self.binary_columns.append(column)
                else:
                    self.categorical_columns[column] = dataframe[column].mode()
                    if cardinality > self.max_cardinality:
                        self.need_freq_encoding.add(column)
            elif np.issubdtype(datatype, np.integer) and dataframe[column].nunique() == 2:
                self.binary_columns.append(column)
            else:
                self.numerical_columns.append(column)

        self._column_order = self.binary_columns + list(self.categorical_columns.keys()) + self.numerical_columns

        # fit encoders
        encoders = dict()
        for column in self.binary_columns:
            encoders[column] = LabelEncoder().fit(dataframe[column].astype(str))

        for column in self.categorical_columns:
            if column in self.need_freq_encoding:
                encoders[column] = FreqLabelEncoder().fit(dataframe[column].astype(str))
            else:
                encoders[column] = LabelEncoder().fit(dataframe[column].astype(str))
            self._cards.append(len(encoders[column]))

        for column in self.numerical_columns:
            encoders[column] = StandardScaler().fit(dataframe[column])

        self._embeds = [int(min(600, 1.6 * card ** .5)) for card in self._cards]
        self.encoders = encoders
        return self

    def transform(self, dataframe):
        df = dataframe[self._column_order].copy()
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                df[column] = encoder.transform(df[column])
            else:
                df[column] = encoder.transform(df[column].astype(str))
        return df.values

    def transform_single(self, x):
        output = []
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                output.append(encoder.transform([x[column]])[0])
            else:
                output.append(encoder.transform([str(x[column])])[0])
        return np.array(output)

    @property
    def n_bins(self): return len(self.binary_columns)

    @property
    def n_cats(self): return len(self.categorical_columns)

    @property
    def n_nums(self): return len(self.numerical_columns)

    @property
    def cards(self): return self._cards

    @property
    def embeds(self): return self._embeds

    def datatype_info(self): return {'n_bins': self.n_bins, 'n_cats': self.n_cats, 'n_nums': self.n_nums}


class SingleDataset(Dataset):
    def __init__(self, data, datatype_info):
        self.data = data
        self.n_bins = datatype_info['n_bins']
        self.n_cats = datatype_info['n_cats']
        self.n_nums = datatype_info['n_nums']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        single_data = dict()
        if self.n_bins: single_data['bins'] = self.data[index, :self.n_bins].astype('float32')
        if self.n_cats: single_data['cats'] = self.data[index, self.n_bins: self.n_bins + self.n_cats].astype('float32')
        if self.n_nums: single_data['nums'] = self.data[index, -self.n_nums:].astype('float32')
        return single_data