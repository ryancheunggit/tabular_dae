import numpy as np
from operator import lt, gt


class AverageMeter(object):
    def __init__(self, mv_size=128, init_val=None):
        self.mv_size = mv_size
        self.reset(init_val)

    def reset(self, init_val=None):
        self.val = [] if not init_val else [init_val]

    def update(self, val):
        self.val.append(val)

    @property
    def avg(self):
        return np.mean(self.val[-self.mv_size:])

    @property
    def overall_avg(self):
        return np.mean(self.val)


class EarlyStopping(object):
    """Monitoring an metric, flag when to stop training."""
    def __init__(self, mode='min', min_delta=0, percentage=False, patience=10, initial_bad=0, initial_best=np.nan, verbose=0):
        assert patience > 0, 'patience must be positive integer'
        assert mode in ['min', 'max'], 'mode must be either min or max'
        self.mode = mode
        self.patience = patience
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = initial_bad
        self.is_better = self._init_is_better(mode, min_delta, percentage)
        self.verbose = verbose
        self._stop = False

    def step(self, metric):
        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if np.isnan(self.best) and (not np.isnan(metric)):
            self.num_bad_epochs = 0
            self.best = metric

        self._stop = self.num_bad_epochs >= self.patience
        if self.verbose and self._stop: print('Early Stopping Triggered, best score is: ', self.best)
        return self._stop

    def _init_is_better(self, mode, min_delta, percentage):
        comparator = lt if mode == 'min' else gt
        if not percentage:
            def _is_better(new, best):
                target = best - min_delta if mode == 'min' else best + min_delta
                return comparator(new, target)
        else:
            def _is_better(new, best):
                target = best * (1 - (min_delta / 100)) if mode == 'min' else best * (1 + (min_delta / 100))
                return comparator(new, target)
        return _is_better

    def __repr__(self):
        return '<EarlyStopping object with: mode - {} - num_bad_epochs - {} - patience - {} - best - {}>'.format(
            self.mode, self.num_bad_epochs, self.patience, self.best)
