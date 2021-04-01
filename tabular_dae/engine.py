import torch
import numpy as np
from torch.utils.data import DataLoader
from .network import AutoEncoder, SwapNoiseCorrupter
from .data import SingleDataset
from .util import AverageMeter, EarlyStopping


def _init_dataloaders(data, datatype_info, batch_size, validation_ratio):
    ''' Split data into training set and validation set, construct dataloader for each. '''
    n = len(data)
    cutoff = int(n * (1 - validation_ratio))

    train_ds = SingleDataset(data[:cutoff, :], datatype_info)
    valid_ds = SingleDataset(data[cutoff:, :], datatype_info)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return train_dl, valid_dl


def _init_swap_noise_makers(datatype_info, swap_noise_probas):
    ''' Create swap noise corrupter for each input data type. '''
    n_bins, n_cats, n_nums = datatype_info['n_bins'], datatype_info['n_cats'], datatype_info['n_nums']
    if isinstance(swap_noise_probas, (float, int)): swap_noise_probas = [swap_noise_probas] * sum([n_bins, n_cats, n_nums])

    noise_makers = dict()
    if n_bins: noise_makers['bins'] = SwapNoiseCorrupter(swap_noise_probas[:n_bins])
    if n_cats: noise_makers['cats'] = SwapNoiseCorrupter(swap_noise_probas[n_bins: n_bins + n_cats])
    if n_nums: noise_makers['nums'] = SwapNoiseCorrupter(swap_noise_probas[-n_nums:])
    return noise_makers


def _apply_noise(batch_data, noise_makers):
    ''' Apply swap noise on data. '''
    noisy_batch, masks = dict(), dict()
    for typ in ['bins', 'cats', 'nums']:
        if typ in batch_data:
            noisy_data, mask = noise_makers[typ](batch_data[typ])
            noisy_batch[typ] = noisy_data
            masks[typ] = mask
    return noisy_batch, masks


def _init_loss_weights(loss_weights, datatype_info, mask_loss_weight=2):
    if isinstance(loss_weights, dict): return loss_weights
    n_bins, n_cats, n_nums = datatype_info['n_bins'], datatype_info['n_cats'], datatype_info['n_nums']
    loss_weights = dict()
    total = sum([n_bins, n_cats, n_nums])
    if n_bins: loss_weights['bins'] = n_bins / total
    if n_cats: loss_weights['cats'] = n_cats / total
    if n_nums: loss_weights['nums'] = n_nums / total
    loss_weights['mask'] = mask_loss_weight
    return loss_weights


def train(network_cfg_or_network,
          data,
          datatype_info,
          swap_noise_probas,
          validation_ratio,   # TODO: Any good reason to allow training without validation split?
          batch_size=128,
          max_epochs=1024,
          early_stopping_rounds=100,
          eval_verbose=10,
          verbose=2,
          optimizer_fn=torch.optim.Adam,
          optimizer_params={'lr': 3e-4},
          scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
          scheduler_params=dict(),
          loss_weights=None,
          mask_loss_weight=2,
          device='cpu',
          model_checkpoint='./model_checkpoint.pth'
    ):
    # make sure network is ready and on device.
    assert isinstance(network_cfg_or_network, (dict, AutoEncoder)), 'either supply network itself, or a recepit to make one.'
    if isinstance(network_cfg_or_network, dict):
        network = AutoEncoder(**network_cfg_or_network).to(device)
    else:
        network = network_cfg_or_network.to(device)

    # prepare to train
    train_dl, valid_dl = _init_dataloaders(data, datatype_info, batch_size, validation_ratio)
    noise_makers = _init_swap_noise_makers(datatype_info, swap_noise_probas)
    loss_weights = _init_loss_weights(loss_weights, datatype_info, mask_loss_weight)
    optimizer = optimizer_fn(network.parameters(), **optimizer_params)
    scheduler = None if not scheduler_fn else scheduler_fn(optimizer, verbose=verbose, **scheduler_params)
    earlystop = None if early_stopping_rounds == 0 else EarlyStopping(patience=early_stopping_rounds, verbose=verbose)

    network = network.to(device)

    best_score = float('inf')
    # training network
    for epoch in range(max_epochs):

        # train step
        network.train()
        meter = AverageMeter()
        for i, x in enumerate(train_dl):
            for k in x: x[k] = x[k].to(device, non_blocking=True)
            noisy_x, masks = _apply_noise(x, noise_makers)
            optimizer.zero_grad()
            reconstruction, predicted_mask = network(noisy_x)
            loss = network.loss(x, masks, reconstruction, predicted_mask, loss_weights)
            loss.backward(); optimizer.step()
            meter.update(loss.item())
            if verbose > 1:
                print('\repoch {:4d} - batch {:4d} - train loss {:6.4f}'.format(epoch, i, meter.avg), end='')
        train_loss = meter.overall_avg

        # validation step
        meter.reset()
        with torch.no_grad():
            for i, x in enumerate(valid_dl):
                for k in x: x[k] = x[k].to(device, non_blocking=True)
                noisy_x, masks = _apply_noise(x, noise_makers)
                reconstruction, predicted_mask = network(noisy_x)
                loss = network.loss(x, masks, reconstruction, predicted_mask, loss_weights)
                meter.update(loss.item())
                if verbose > 1:
                    print('\repoch {:4d} - batch {:4d} - valid loss {:6.4f}'.format(epoch, i, meter.avg), end='')
        valid_loss = meter.overall_avg

        if verbose and epoch % eval_verbose == 0:
            print('\repoch {:4d} - train loss {:6.4f} - valid loss {:6.4f}'.format(epoch, train_loss, valid_loss))

        # adjust learning rate if neccessary
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step(epoch)

        # checkpointing
        if valid_loss < best_score:
            best_score = valid_loss
            torch.save({'model': network.state_dict()}, model_checkpoint)

        # early stopping
        if earlystop.step(valid_loss):
            break

    # retrieve the best weights
    model_state_dict = torch.load(model_checkpoint)
    network.load_state_dict(model_state_dict['model'])
    return network


def featurize(network, data, datatype_info, batch_size, device='cpu'):
    ds = SingleDataset(data, datatype_info)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    features = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            for k in x: x[k] = x[k].to(device, non_blocking=True)
            batch_featurs = network.featurize(x)
            features.append(batch_featurs.detach().cpu().numpy())
    features = np.vstack(features)
    return features