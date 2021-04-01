from .data import DataFrameParser
from .engine import train, featurize


class DAE(object):
    def __init__(
            self,
            body_network='deepstack',
            body_network_cfg=dict(hidden_size=128),
            swap_noise_probas=.2,
            cats_handling='onehot',
            cards=[],
            embeded_dims=[],
            device='cpu'
        ):
        super().__init__()
        self.body_network = body_network
        self.body_network_cfg = body_network_cfg
        self.swap_noise_probas = swap_noise_probas
        self.cats_handling = cats_handling
        self.cards = cards
        self.embeded_dims = embeded_dims
        self.device = device
        self.network = None

    def _parse_dataframe(self, dataframe):
        self.parser = parser = DataFrameParser().fit(dataframe)
        self.datatype_info = datatype_info = parser.datatype_info()
        data = parser.transform(dataframe)
        if not self.cards: self.cards = parser.cards
        if not self.embeded_dims: self.embeded_dims = parser.embeds
        return data, datatype_info

    def fit(
            self,
            dataframe,
            batch_size=32,
            max_epochs=1024,
            validation_ratio=.2,
            early_stopping_rounds=50,
            verbose=2,
            **train_kwargs
        ):
        data, datatype_info = self._parse_dataframe(dataframe)

        network_cfg = dict(
            datatype_info=datatype_info,
            cards=self.cards,
            embeded_dims=self.embeded_dims,
            cats_handling=self.cats_handling,
            body_network=self.body_network,
            body_network_cfg=self.body_network_cfg,
        )
        self.network = train(
            network_cfg,
            data,
            datatype_info,
            self.swap_noise_probas,
            batch_size=batch_size,
            validation_ratio=validation_ratio,
            early_stopping_rounds=early_stopping_rounds,
            max_epochs=max_epochs,
            device=self.device,
            verbose=verbose,
            **train_kwargs
        )

    def transform(self, dataframe, batch_size=32):
        assert self.network is not None, 'model not trained yet'
        return featurize(
            self.network,
            self.parser.transform(dataframe),
            self.datatype_info,
            batch_size=batch_size,
            device=self.device
        )