import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange


__all__ = ['SwapNoiseCorrupter', 'AutoEncoder']


class SwapNoiseCorrupter(nn.Module):
    """
        Apply swap noise on the input data.

        Each data point has specified chance be replaced by a random value from the same column.
    """
    def __init__(self, probas):
        super().__init__()
        self.probas = torch.from_numpy(np.array(probas))

    def forward(self, x):
        should_swap = torch.bernoulli(self.probas.to(x.device) * torch.ones((x.shape)).to(x.device))
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x, mask


class EntityEmbedder(nn.Module):
    ''' Embed categorical variables. '''
    def __init__(self, cards, embeded_dims):
        super().__init__()
        assert len(cards) == len(embeded_dims), 'require embedding dim for each categorical variables'
        self.embedders = nn.ModuleList([
            nn.Embedding(c, s)
            for c, s in zip(cards, embeded_dims)
        ])

    def forward(self, x):
        return torch.cat([
            embedder(x[:, i].long())
            for i, embedder in enumerate(self.embedders)
        ], dim=1)


class OneHotEncoder(nn.Module):
    ''' One-Hot encode categorical variables. '''
    def __init__(self, cards):
        super().__init__()
        self.cards = cards

    def forward(self, x):
        return torch.cat([
            F.one_hot(x[:, i].long(), num_classes=num_classes)
            for i, num_classes in enumerate(self.cards)
        ], dim=1)


def _make_mlp_layers(num_units):
    layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
        for in_features, out_features in zip(num_units, num_units[1:])
    ])
    return layers


class DeapStack(nn.Module):
    ''' Simple MLP body. '''
    def __init__(self, in_features, hidden_size, num_layers=3):
        super().__init__()
        self.layers = _make_mlp_layers([in_features] + [hidden_size] * num_layers)
        self._output_shape = hidden_size

    @property
    def output_shape(self): return self._output_shape

    def forward_pass(self, x):
        outputs = []
        for layer in self.layers:
            x = F.relu(layer(x))
            outputs.append(x)
        return outputs

    def forward(self, x):
        return self.forward_pass(x)[~0]

    def featurize(self, x):
        return torch.cat(self.forward_pass(x), dim=1)


class DeepBottleneck(nn.Module):
    ''' Simple MLP body with bottleneck. '''
    def __init__(self, in_features, hidden_size, bottleneck_size, num_layers=3):
        super().__init__()
        encoder_layers = num_layers >> 1
        decoder_layers = num_layers - encoder_layers - 1
        self.encoder = _make_mlp_layers([in_features] + [hidden_size] * encoder_layers)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.decoder = _make_mlp_layers([bottleneck_size] + [hidden_size] * decoder_layers)
        self._output_shape = hidden_size

    @property
    def output_shape(self): return self._output_shape

    def forward_pass(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
        x = b = self.bottleneck(x)
        for layer in self.decoder:
            x = F.relu(layer(x))
        return [b, x]

    def forward(self, x):
        return self.forward_pass(x)[1]

    def featurize(self, x):
        return self.forward_pass(x)[0]


class TransformerEncoder(nn.Module):
    ''' Transformer Encoder. '''
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be a multiple of num_heads'
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        ''' input is of shape num_subspaces x batch_size x embed_dim '''
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(F.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x


class Transformer(nn.Module):
    ''' DAE Body with transformer encoders. '''
    def __init__(self, in_features, hidden_size=1024, num_subspaces=8, embed_dim=128, num_heads=8, dropout=0, feedforward_dim=512, num_layers=3):
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces, 'num_subspaces must be a multiple of embed_dim'
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

        self.excite = nn.Linear(in_features, hidden_size)
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
            for _ in range(num_layers)
        ])
        self._output_shape = hidden_size

    @property
    def output_shape(self): return self._output_shape

    def divide(self, x):
        return rearrange(x, 'b (s e) -> s b e', s=self.num_subspaces, e=self.embed_dim)

    def combine(self, x):
        return rearrange(x, 's b e -> b (s e)')

    def forward_pass(self, x):
        x = F.relu(self.excite(x))
        x = self.divide(x)
        outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            outputs.append(x)
        return outputs

    def forward(self, x):
        return self.combine(self.forward_pass(x)[~0])

    def featurize(self, x):
        return torch.cat([self.combine(x) for x in self.forward_pass(x)], dim=1)


class MultiTaskHead(nn.Module):
    """
        Simple Linear transformation to take last hidden representation to reconstruct inputs.

        Output is dictionary of variable type to tensor mapping.
    """
    def __init__(self, in_features, n_bins=0, n_cats=0, n_nums=0, cards=[]):
        super().__init__()
        assert n_cats == len(cards), 'require cardinalities for each categorical variable'
        assert n_bins + n_cats + n_nums, 'need some targets'
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums

        self.bins_linear = nn.Linear(in_features, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(in_features, card) for card in cards])
        self.nums_linear = nn.Linear(in_features, n_nums) if n_nums else None

    def forward(self, features):
        outputs = dict()

        if self.bins_linear:
            outputs['bins'] = self.bins_linear(features)

        if self.cats_linears:
            outputs['cats'] = [linear(features) for linear in self.cats_linears]

        if self.nums_linear:
            outputs['nums'] = self.nums_linear(features)

        return outputs


_ae_body_options = {
    'deepstack': DeapStack,
    'deepbottleneck': DeepBottleneck,
    'transformer': Transformer
}


class AutoEncoder(nn.Module):
    ''' AutoEncoder. '''
    def __init__(self, datatype_info, cards=[], embeded_dims=[], cats_handling='onehot',
                       body_network=DeapStack, body_network_cfg=dict()):
        super().__init__()
        self.n_bins = n_bins = datatype_info.get('n_bins', 0)
        self.n_cats = n_cats = datatype_info.get('n_cats', 0)
        self.n_nums = n_nums = datatype_info.get('n_nums', 0)
        self.cards = cards
        self.embeded_dims = embeded_dims if isinstance(embeded_dims, (list, tuple, np.ndarray)) else [embeded_dims] * n_cats

        if isinstance(body_network, str):
            body_network = _ae_body_options[body_network]

        self.body_network_in_features = n_bins + n_nums

        self.cats_handler = None
        if n_cats:
            if cats_handling == 'onehot':
                self.cats_handler = OneHotEncoder(cards)
                self.body_network_in_features += sum(cards)
            if cats_handling == 'embed':
                self.cats_handler = EntityEmbedder(cards, self.embeded_dims)
                self.body_network_in_features += sum(self.embeded_dims)

        self.body = body_network(in_features=self.body_network_in_features, **body_network_cfg)
        self.reconstruction_head = MultiTaskHead(self.body.output_shape, n_bins, n_cats, n_nums, cards)
        self.mask_predictor_head = nn.Linear(self.body.output_shape, sum([n_bins, n_cats, n_nums]))

    def process_inputs(self, inputs):
        """
            Process inputs to the network.

            Input is dictionary of variable type to tensor mapping.
            Output is 2D tensor of size: batch_size x body_network_in_features.
        """
        x_bins = inputs.get('bins', None)
        x_cats = inputs.get('cats', None)
        x_nums = inputs.get('nums', None)
        if x_cats is not None: x_cats = self.cats_handler(x_cats)
        body_network_inputs = torch.cat([t for t in [x_bins, x_cats, x_nums] if t is not None], dim=1)
        return body_network_inputs

    def forward(self, inputs):
        """ Forward pass of AutoEncoder network.

            Inputs is dictionary of variable type to tensor mapping.
            Outputs are tuple of reconstruction of inputs and prediction of corruption mask.
        """
        body_network_inputs = self.process_inputs(inputs)
        last_hidden = self.body(body_network_inputs)
        reconstruction = self.reconstruction_head(last_hidden)
        predicted_mask = self.mask_predictor_head(last_hidden)
        return reconstruction, predicted_mask

    def featurize(self, inputs):
        body_network_inputs = self.process_inputs(inputs)
        features = self.body.featurize(body_network_inputs)
        return features


    def loss(self, inputs, masks, reconstruction, predicted_mask, loss_weights):
        """ Calculating the loss for DAE network.

            BCE for masks and reconstruction of binary inputs.
            CE for categoricals.
            MSE for numericals.

            reconstruction loss is weighted average of mean reduction of loss per datatype.
            mask loss is mean reduced.
            final loss is weighted sum of reconstruction loss and mask loss.
        """
        flattened_masks = torch.cat(list(masks.values()), dim=1)
        mask_loss = loss_weights['mask'] * F.binary_cross_entropy_with_logits(predicted_mask, flattened_masks)

        reconstruction_losses = dict()
        if 'bins' in inputs:
            reconstruction_losses['bins'] = loss_weights['bins'] * F.binary_cross_entropy_with_logits(reconstruction['bins'], inputs['bins'])

        if 'cats' in inputs:
            cats_losses = []
            for i in range(len(reconstruction['cats'])):
                cats_losses.append(F.cross_entropy(reconstruction['cats'][i], inputs['cats'][:, i].long()))
            reconstruction_losses['cats'] = loss_weights['cats'] * torch.stack(cats_losses).mean()

        if 'nums' in inputs:
            reconstruction_losses['nums'] = loss_weights['nums'] * F.mse_loss(reconstruction['nums'], inputs['nums'])

        reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()
        return reconstruction_loss + mask_loss
