from __future__ import division
from itertools import izip, islice
from operator import add
import math

import numpy as np


def signal_to_freq_bins(y, cutoffs, sampling_rate=512.0, window='hanning'):
    # prevent spectral leakage
    if window == 'hanning':
        window = np.hanning(len(y))
        y = window[:, np.newaxis] * y

    Y = np.fft.fft(y)
    f = abs(Y)
    #freqs = np.fft.fftfreq(len(y), d=1/sampling_rate)
    #cutoffs2 = []
    #for c in cutoffs:
    #    for freq_i, freq in enumerate(freqs):
    #        if freq > c:
    #            cutoffs2.append(freq_i)
    #            break
    cutoffs_scaled = [c * len(y) / sampling_rate for c in cutoffs]

    # need a vectorlike zero element for bins to have the same dimensions
    bins = [reduce(add, f[math.ceil(low):math.floor(high)], np.zeros(f.shape[1:])) for low, high
            in izip(cutoffs_scaled, islice(cutoffs_scaled, 1, None))]
    return bins

if __name__ == "__main__":
    import numpy
    eeg = "57 38 20 29 56 54 35 21 22 13 -7 -26 -30 -8 10 4 -10 -5 16 33 49 55 43 41 39 32 28 44 32 -6 -22 0 37 67 50 21 20 27 23 20 22 22 25 25 38 59 45 8 7 22 7 -21 -27 -35 -37 -11 23 53 84 100 86 71 66 55 41 34 27 13 12 65 118 120 105 98 67 44 55 41 6 -10 -6 3 11 33 81 121 120 100 88 98 107 112 117 115 112 123 145 156 150 135 134 134 108 72 52 52 53 42 25 18 22 33 53 58 45 36 44 68 77 88 102 105 121 137 136 133 145 168 201 198 155 135 149 195 220 205 181 165 168 176 164 129 97 84 87 100 106 113 116 101 76 93 149 187 195 209 240 283 324 326 299 281 293 321 338 332 336 355 354 320 290 268 245 219 232 244 186 136 137 161 171 163 153 145 135 155 179 170 140 117 137 169 173 179 177 170 173 167 149 135 147 154 147 157 183 193 195 187 179 169 136 113 117 136 132 103 99 129 154 156 154 154 161 164 161 149 144 165 186 193 195 172 120 74 66 61 48 24 26 52 70 75 87 114 113 90 70 60 59 82 120 120 67 22 5 -6 1 10 23 38 33 23 32 49 33 5 4 16 26 21 18 17 -4 -21 -9 1 -10 -26 -23 -13 -13 -19 -26 -18 10 32 12 -13 -8 26 52 57 59 54 57 80 102 118 136 170 192 201 204 194 187 209 218 203 171 135 120 123 138 137 115 100 92 105 119 98 35 -12 -24 -17 -8 -29 -59 -66 -49 -72 -121 -135 -117 -116 -140 -166 -154 -121 -92 -85 -87 -86 -75 -81 -88 -81 -86 -93 -78 -56 -52 -42 -12 -3 5 45 65 18 -29 -21 23 50 43 44 64 75 73 64 59 44 26 28 53 70 45 5 -22 -20 12 23 -1 -36 -38 -28 -27 -27 -22 -9 7 12 12 21 39 66 74 74 83 97 117 141 147 137 140 153 162 135 86 65 83 103 99 67 39 25 21 37 52 56 66 76 87 92 84 68 68 86 106 108 101 102 102 105 109 112 115 109 85 50 29 18 10 22 23 1 -19 -30 -38 -38 -38 -26 -4 4 -1 7 28 26 16 19 23 -5 -35 -39 -11 16 8 -3 -5 -18 -26 -9 19 36 33 37 52 55 38 25 37 58 83 103 120 120 118 123 116 91 72 67 49 10 -2 4 27 52 57 58 55 51 36 11 7 24 33 22 11"
    eeg = numpy.array([float(v) for v in eeg.split(' ')], dtype='complex128')
    signal_to_freq_bins(eeg, cutoffs=[0.5, 4.0, 7.0, 12.0, 30.0], sampling_rate=512.0)
