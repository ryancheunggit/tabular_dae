import os
import torch
import pandas as pd
from tabular_dae.data import DataFrameParser
from sklearn.linear_model import RidgeClassifierCV
from tabular_dae.engine import train, featurize

def test_on_titanic():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data', 'titanic.csv'))
    y = df['Survived']
    df.drop('Survived', axis=1, inplace=True)

    parser = DataFrameParser().fit(df)
    data = parser.transform(df)
    datatype_info = parser.datatype_info()

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    batch_size = 32
    validation_ratio = .2
    swap_noise_probas = [.1] + [.25] * 4 + [.25] * 6

    network_cfgs = {
        'deepstack': dict(
            datatype_info=datatype_info,
            cards=parser.cards,
            embeded_dims=[],
            cats_handling='onehot',
            body_network='deepstack',
            body_network_cfg=dict(hidden_size=128),
        ),
        'deepbottleneck': dict(
            datatype_info=datatype_info,
            cards=parser.cards,
            embeded_dims=parser.embeds,
            cats_handling='embed',
            body_network='deepbottleneck',
            body_network_cfg=dict(
                hidden_size=512,
                bottleneck_size=16
            ),
        ),
        'transformer': dict(
            datatype_info=datatype_info,
            cards=parser.cards,
            embeded_dims=[],
            cats_handling='onehot',
            body_network='transformer',
            body_network_cfg=dict(
                hidden_size=128,
                num_subspaces=4,
                embed_dim=32,
                num_heads=4,
                dropout=0,
                feedforward_dim=64
            ),
        )
    }

    for network_type, network_cfg in network_cfgs.items():
        network = train(
            network_cfg,
            data,
            datatype_info,
            swap_noise_probas,
            batch_size=batch_size,
            validation_ratio=validation_ratio,
            max_epochs=256,
            device=device,
            verbose=1
        )
        features = featurize(network, data, datatype_info, batch_size=batch_size*2, device=device)
        classifier = RidgeClassifierCV(cv=5).fit(features, y)
        assert classifier.best_score_ > .78