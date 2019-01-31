from tqdm import tqdm


class ProgressBar(object):

    def __init__(self, start_t, stop_t, show=True, label=None):
        self.t = start_t
        if show:
            self.tqdm = tqdm(initial=start_t, total=stop_t, desc=label,
                             unit='s (sim)', unit_scale=True)
        else:
            self.tqdm = None

    def update(self, t):
        if self.tqdm is not None:
            self.tqdm.update(t - self.t)
            self.t = t
