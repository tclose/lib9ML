import math
from progressbar import ProgressBar


def create_progress_bar(start, stop, dt):
    progress_bar = ProgressBar(start, stop)
    progress_bar.res = -int(math.floor(math.log10(dt)))
    return progress_bar
