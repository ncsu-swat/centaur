
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class WindowTuple(tuple):
    def __len__(self):
        return 0


def istft_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D signal STFT, default axes, onesided
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (
        np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(1.0, dtype=np.float32),
        "window": WindowTuple(("hann",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two-sided STFT, float64 complex
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft
    ntime = 8
    Zxx = (
        np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
    ).astype(np.complex128)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(16000.0, dtype=np.float64),
        "window": WindowTuple(("hamming",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Swapped axes (time_axis=-2, freq_axis=-1)
    nperseg = 32
    noverlap = 16
    nfft = 32
    nfreq = nfft // 2 + 1
    ntime = 12
    Zxx = (
        np.random.randn(ntime, nfreq) + 1j * np.random.randn(ntime, nfreq)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(44100.0, dtype=np.float32),
        "window": WindowTuple(("tukey", 0.25)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -2,
        "freq_axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched input, nfft > nperseg
    batch_size = 3
    nperseg = 8
    noverlap = 4
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 6
    Zxx = (
        np.random.randn(batch_size, nfreq, ntime)
        + 1j * np.random.randn(batch_size, nfreq, ntime)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(8000.0, dtype=np.float32),
        "window": WindowTuple(("blackman",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: boundary=False with Kaiser window
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (
        np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(100.0, dtype=np.float32),
        "window": WindowTuple(("kaiser", 3.0)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 75% overlap with Gaussian window
    nperseg = 16
    noverlap = 12
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 14
    Zxx = (
        np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
    ).astype(np.complex128)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(500.0, dtype=np.float64),
        "window": WindowTuple(("gaussian", 2.0)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched input, two-sided STFT, boxcar window
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft
    ntime = 8
    Zxx = (
        np.random.randn(2, 2, nfreq, ntime)
        + 1j * np.random.randn(2, 2, nfreq, ntime)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(1.0, dtype=np.float32),
        "window": WindowTuple(("boxcar",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive axis indices (time_axis=1, freq_axis=2) in 3D array
    batch_size = 2
    nperseg = 8
    noverlap = 4
    nfft = 8
    nfreq = nfft // 2 + 1
    ntime = 9
    Zxx = (
        np.random.randn(batch_size, ntime, nfreq)
        + 1j * np.random.randn(batch_size, ntime, nfreq)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(2.0, dtype=np.float32),
        "window": WindowTuple(("bartlett",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": 1,
        "freq_axis": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive axis indices for 2D array (time_axis=0, freq_axis=1), two-sided
    nperseg = 8
    noverlap = 4
    nfft = 8
    nfreq = nfft
    ntime = 7
    Zxx = (
        np.random.randn(ntime, nfreq) + 1j * np.random.randn(ntime, nfreq)
    ).astype(np.complex64)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(10.0, dtype=np.float32),
        "window": WindowTuple(("bohman",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": False,
        "time_axis": 0,
        "freq_axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger window and FFT with cosine window
    nperseg = 64
    noverlap = 32
    nfft = 64
    nfreq = nfft // 2 + 1
    ntime = 16
    Zxx = (
        np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
    ).astype(np.complex128)
    input_dict = {
        "Zxx": Zxx,
        "fs": np.array(22050.0, dtype=np.float64),
        "window": WindowTuple(("cosine",)),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.istft_5"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_5'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_5'], lib="jax", suffix=5)
