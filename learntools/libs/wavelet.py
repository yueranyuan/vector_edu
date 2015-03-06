import numpy as np
import pywt

import matplotlib.pyplot as plt


def _downsample(arr, n):
    """Downsample a signal by averaging neighboring points.

    Code adapted from
    http://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array"""
    end = n * int(len(arr) / n)
    return np.std(arr[:end].reshape(n, -1), 1)

# Make a scalogram given an MRA tree.
def scalogram(data):
    bottom = 0

    vmin = min(map(lambda x: min(abs(x)), data))
    vmax = max(map(lambda x: max(abs(x)), data))

    plt.gca().set_autoscale_on(False)

    for row in range(0, len(data)):
        scale = 2.0 ** (row - len(data))

        plt.imshow(
            np.array([abs(data[row])]),
            interpolation = 'nearest',
            vmin = vmin,
            vmax = vmax,
            extent = [0, 1, bottom, bottom + scale])

        bottom += scale


def signal_to_wavelet(y, family='db2', double=False, min_length=10, max_length=None, depth=1):
    c_a, c_d = pywt.dwt(y, family)

    # determine whether we want to split the approximation and/or the detail any further
    if depth <= 1:
        split_a = False
        split_d = False
    elif double:
        split_a = True
        split_d = True
    else:
        split_a = True
        split_d = False

    # a function to recursively split the wave or downsample it
    def _prepare_wave(to_split, wave):
        if to_split:
            return signal_to_wavelet(wave, family=family, min_length=min_length,
                                     max_length=max_length, depth=depth - 1, double=double)
        return [_downsample(wave, max_length)]

    a = _prepare_wave(split_a, c_a)
    d = _prepare_wave(split_d, c_d)
    return a + d