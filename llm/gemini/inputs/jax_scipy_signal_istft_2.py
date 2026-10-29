
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class WindowSpec(tuple):

    def __array__(self, *args, **kwargs):
        return np.array([0.0, 1.0], dtype=np.float64)


def istft_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D complex64, Hann window, onesided
    nperseg = 16
    noverlap = 8
    nfft = 16
    n_bins = nfft // 2 + 1
    n_frames = 10
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 1.0,
        "window": WindowSpec(("hann",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Hamming window, complex128, boundary=False, fs=16000.0
    nperseg = 32
    noverlap = 16
    nfft = 32
    n_bins = nfft // 2 + 1
    n_frames = 12
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex128)
    input_dict = {
        "Zxx": zxx,
        "fs": 16000.0,
        "window": WindowSpec(("hamming",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two-sided STFT (input_onesided=False), Boxcar window
    nperseg = 16
    noverlap = 8
    nfft = 16
    n_bins = nfft
    n_frames = 8
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 44100.0,
        "window": WindowSpec(("boxcar",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: nfft > nperseg, Tukey window with parameter
    nperseg = 16
    noverlap = 8
    nfft = 32
    n_bins = nfft // 2 + 1
    n_frames = 10
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 8000.0,
        "window": WindowSpec(("tukey", 0.25)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D input with batch dimension at index 0, Blackman window
    batch_size = 3
    nperseg = 16
    noverlap = 8
    nfft = 16
    n_bins = nfft // 2 + 1
    n_frames = 10
    zxx = (
        np.random.randn(batch_size, n_bins, n_frames)
        + 1j * np.random.randn(batch_size, n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 22050.0,
        "window": WindowSpec(("blackman",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": 2,
        "freq_axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Transposed axes: time_axis=0, freq_axis=1, Bartlett window
    nperseg = 8
    noverlap = 4
    nfft = 8
    n_bins = nfft // 2 + 1
    n_frames = 10
    zxx = (
        np.random.randn(n_frames, n_bins)
        + 1j * np.random.randn(n_frames, n_bins)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 100.0,
        "window": WindowSpec(("bartlett",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": 0,
        "freq_axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Kaiser window with beta parameter, high overlap (75%)
    nperseg = 16
    noverlap = 12
    nfft = 16
    n_bins = nfft // 2 + 1
    n_frames = 15
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex128)
    input_dict = {
        "Zxx": zxx,
        "fs": 2.5,
        "window": WindowSpec(("kaiser", 4.0)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Gaussian window, two-sided, nfft > nperseg
    nperseg = 32
    noverlap = 24
    nfft = 64
    n_bins = nfft
    n_frames = 12
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 500.0,
        "window": WindowSpec(("gaussian", 2.0)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D input with batch at the last dimension
    batch_size = 4
    nperseg = 16
    noverlap = 8
    nfft = 16
    n_bins = nfft // 2 + 1
    n_frames = 10
    zxx = (
        np.random.randn(n_bins, n_frames, batch_size)
        + 1j * np.random.randn(n_bins, n_frames, batch_size)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 12000.0,
        "window": WindowSpec(("hann",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": 1,
        "freq_axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small segment size, Boxcar window
    nperseg = 4
    noverlap = 2
    nfft = 4
    n_bins = nfft // 2 + 1
    n_frames = 6
    zxx = (
        np.random.randn(n_bins, n_frames)
        + 1j * np.random.randn(n_bins, n_frames)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 10.0,
        "window": WindowSpec(("boxcar",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative axes with time_axis=-2, freq_axis=-1
    nperseg = 16
    noverlap = 8
    nfft = 16
    n_bins = nfft // 2 + 1
    n_frames = 12
    zxx = (
        np.random.randn(n_frames, n_bins)
        + 1j * np.random.randn(n_frames, n_bins)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": zxx,
        "fs": 1000.0,
        "window": WindowSpec(("hamming",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": -2,
        "freq_axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.istft_2"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_2'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_2'], lib="jax", suffix=2)
