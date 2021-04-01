import torch
import numpy as np
from tabular_dae.network import (
    EntityEmbedder,
    OneHotEncoder,
    SwapNoiseCorrupter,
    DeapStack,
    DeepBottleneck,
    TransformerEncoder,
    Transformer,
    MultiTaskHead,
    AutoEncoder
)


def test_entity_embedder():
    x = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [2, 2, 0],
        [3, 0, 1],
        [5, 2, 1],
        [4, 0, 2]
    ])
    cards = x.max(0) + 1
    embeded_dims = np.floor(np.sqrt(cards)).astype('int')
    embedder = EntityEmbedder(cards, embeded_dims)
    embeded = embedder(torch.from_numpy(x))
    assert embeded.shape[1] == sum(embeded_dims)


def test_onehot_encoder():
    x = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [2, 2, 0],
        [3, 0, 1],
        [5, 2, 1],
        [4, 0, 2]
    ])
    cards = [8, 5, 5]
    encoder = OneHotEncoder(cards)
    encoded = encoder(torch.from_numpy(x))
    assert encoded.shape[1] == sum(cards)


def test_swap_noise():
    probas = [.2, .2, .2, .2, .2]
    m = SwapNoiseCorrupter(probas)
    x_bins = torch.cat([torch.ones(2000, 1), torch.zeros(1200, 1)], dim=0)
    x_cats = torch.cat([torch.ones(150, 1), torch.zeros(50, 1), 2 * torch.ones(3000, 1)], dim=0)
    x_nums = torch.rand((3200, 3))
    x = torch.cat([x_bins, x_cats, x_nums], dim = 1)
    corrupted_x, real_mask = m(x)
    assert corrupted_x.shape == x.shape


def test_deepstack():
    x = torch.rand((32, 5))
    m = DeapStack(5, 20)
    o = m(x)
    f = m.featurize(x)
    assert o.shape == torch.Size([32, 20])
    assert f.shape == torch.Size([32, 60])


def test_deepbottleneck():
    x = torch.rand((32, 5))
    m = DeepBottleneck(5, 20, 3)
    o = m(x)
    f = m.featurize(x)
    assert o.shape == torch.Size([32, 20])
    assert f.shape == torch.Size([32, 3])


def test_tfencoder():
    x = torch.rand((20, 32, 8))
    m = TransformerEncoder(8, 4, .1, 5)
    o = m(x)
    assert o.shape == x.shape


def test_tf():
    x = torch.rand((32, 5))
    m = Transformer(5, 64, 8, 8, 4, .1, 16)
    o = m(x)
    f = m.featurize(x)
    assert o.shape == torch.Size([32, 64])
    assert f.shape == torch.Size([32, 3 * 64])


def test_multitask_head():
    x = torch.rand((32, 128))
    n_bins = 3; n_cats = 2; n_nums = 10; cards = [4, 5]
    m = MultiTaskHead(in_features=128, n_bins=n_bins, n_cats=n_cats, n_nums=n_nums, cards=cards)
    o = m(x)
    assert o['bins'].shape == torch.Size([32, n_bins])
    assert o['cats'][0].shape == torch.Size([32, 4])
    assert o['cats'][1].shape == torch.Size([32, 5])
    assert o['nums'].shape == torch.Size([32, n_nums])

    m = MultiTaskHead(in_features=128, n_cats=n_cats, n_nums=n_nums, cards=cards)
    o = m(x)
    assert 'bins' not in o
    assert o['cats'][0].shape == torch.Size([32, 4])
    assert o['cats'][1].shape == torch.Size([32, 5])
    assert o['nums'].shape == torch.Size([32, n_nums])

    m = MultiTaskHead(in_features=128, n_nums=n_nums)
    o = m(x)
    assert 'bins' not in o
    assert 'cats' not in o
    assert o['nums'].shape == torch.Size([32, n_nums])


def test_ae():
    n_bins = 1; n_cats = 2; n_nums = 5; cards = [4, 5]
    x_bins = torch.cat([torch.ones(2000, 1), torch.zeros(1200, 1)], dim=0)
    x_cats = torch.cat([
        torch.cat([torch.ones(150, 1), torch.zeros(50, 1), 2 * torch.ones(3000, 1)], dim=0),
        torch.cat([torch.ones(700, 1), torch.zeros(1500, 1), 2 * torch.ones(1000, 1)], dim=0)
    ], dim=1).long()
    x_nums = torch.rand((3200, 5))
    x = torch.cat([x_bins, x_cats, x_nums], dim = 1)

    datatype_info = dict(n_bins=n_bins, n_cats=n_cats, n_nums=n_nums)

    m = AutoEncoder(datatype_info, cards, -1, 'onehot', DeapStack, dict(hidden_size=128))
    rec, msk = m({'bins': x_bins, 'cats': x_cats, 'nums': x_nums})
    assert msk.shape == torch.Size([3200, 8])
    assert rec['bins'].shape == torch.Size([3200, 1])
    assert rec['cats'][0].shape == torch.Size([3200, 4])
    assert rec['cats'][1].shape == torch.Size([3200, 5])
    assert rec['nums'].shape == torch.Size([3200, 5])

    m = AutoEncoder(datatype_info, cards, 2, 'embed', DeapStack, dict(hidden_size=128))
    rec, msk = m({'bins': x_bins, 'cats': x_cats, 'nums': x_nums})
    assert msk.shape == torch.Size([3200, 8])
    assert rec['bins'].shape == torch.Size([3200, 1])
    assert rec['cats'][0].shape == torch.Size([3200, 4])
    assert rec['cats'][1].shape == torch.Size([3200, 5])
    assert rec['nums'].shape == torch.Size([3200, 5])