import joblib
from .data import DataFrameParser
from .engine import train, featurize
from .network import AutoEncoder


def load(path_to_model_dump):
    dump = joblib.load(path_to_model_dump)
    model = DAE(**dump['constructor_args'])
    model.parser = dump['parser']
    model.network = AutoEncoder(**model.network_cfg).to(model.device)
    model.network.load_state_dict(dump['network_state_dict'])
    return model


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
        self.parser = None

    def _parse_dataframe(self, dataframe):
        self.parser = parser = DataFrameParser().fit(dataframe)
        data = parser.transform(dataframe)
        if not self.cards: self.cards = parser.cards
        if not self.embeded_dims: self.embeded_dims = parser.embeds
        return data

    @property
    def network_cfg(self):
        return dict(
            datatype_info=self.datatype_info,
            cards=self.cards,
            embeded_dims=self.embeded_dims,
            cats_handling=self.cats_handling,
            body_network=self.body_network,
            body_network_cfg=self.body_network_cfg,
        )

    @property
    def datatype_info(self): return self.parser.datatype_info()

    @property
    def is_fitted(self): return any(val is None for val in [self.network, self.parser])

    def save(self, path_to_model_dump):
        model_state_dict = dict(
            constructor_args=dict(
                body_network=self.body_network,
                body_network_cfg=self.body_network_cfg,
                swap_noise_probas=self.swap_noise_probas,
                cats_handling=self.cats_handling,
                cards=self.cards,
                embeded_dims=self.embeded_dims,
                device=self.device,
            ),
            parser = self.parser,
            network_state_dict=self.network.state_dict()
        )
        joblib.dump(model_state_dict, path_to_model_dump)

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
        data = self._parse_dataframe(dataframe)

        self.network = train(
            network_cfg_or_network=self.network_cfg,
            data=data,
            datatype_info=self.datatype_info,
            swap_noise_probas=self.swap_noise_probas,
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