import pywt


def signal_to_wavelet(y, family='db2', min_length=10, depth=1):
    if depth == 0:
        return []
    c_a, c_d = pywt.dwt(y, family)
    decompose_c_a = signal_to_wavelet(c_a, family=family, min_length=min_length, depth=depth - 1)
    decompose_c_d = signal_to_wavelet(c_d, family=family, min_length=min_length, depth=depth - 1)
    return [c_a] + decompose_c_a + [c_d] + decompose_c_d