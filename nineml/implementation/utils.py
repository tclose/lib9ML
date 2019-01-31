# import math
from tqdm import tqdm


class ProgressBar(object):

    def __init__(self, start_t, stop_t, dt, show=True, label=None):
        self.start_t = start_t
        self.stop_t = stop_t
        self.dt = dt
        print(stop_t)
        if show:
            self.tqdm = tqdm(initial=start_t, total=stop_t, desc=label)
        else:
            self.tqdm = None

    def update(self, t):
        if self.tqdm is not None:
            print(t)
            self.tqdm.update(t)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
