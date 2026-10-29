
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def stft_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like settings
    nperseg = 64
    x = np.random.randn(256).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 1.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 32,
            "nfft": 64,
            "detrend": False,
            "return_onesided": True,
            "boundary": "zeros",
            "padded": True,
            "axis": -1,
        }
    )

    # Input 2: 1D array with zero-padded FFT (nfft > nperseg) and 'even' boundary
    nperseg = 128
    x = np.random.randn(512).astype(np.float64)
    window = np.hamming(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 100.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 64,
            "nfft": 256,
            "detrend": False,
            "return_onesided": True,
            "boundary": "even",
            "padded": True,
            "axis": -1,
        }
    )

    # Input 3: 2D array, computing along axis -1 with 'odd' boundary
    nperseg = 64
    x = np.random.randn(4, 300).astype(np.float32)
    window = np.blackman(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 16000.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 16,
            "nfft": 128,
            "detrend": False,
            "return_onesided": True,
            "boundary": "odd",
            "padded": True,
            "axis": -1,
        }
    )

    # Input 4: 2D array, computing along axis 0 with 'constant' boundary
    nperseg = 32
    x = np.random.randn(200, 8).astype(np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 8000.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 16,
            "nfft": 32,
            "detrend": False,
            "return_onesided": True,
            "boundary": "constant",
            "padded": True,
            "axis": 0,
        }
    )

    # Input 5: 3D array with padded=False
    nperseg = 128
    x = np.random.randn(2, 3, 512).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 44100.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 64,
            "nfft": 128,
            "detrend": False,
            "return_onesided": True,
            "boundary": "zeros",
            "padded": False,
            "axis": -1,
        }
    )

    # Input 6: Complex 1D array with two-sided spectrum (return_onesided=False)
    nperseg = 64
    x = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
    window = np.bartlett(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 2.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 0,
            "nfft": 64,
            "detrend": False,
            "return_onesided": False,
            "boundary": "zeros",
            "padded": True,
            "axis": 0,
        }
    )

    # Input 7: 2D array, return_onesided=False on real input
    nperseg = 50
    x = np.random.randn(5, 200).astype(np.float64)
    window = np.hanning(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 10.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 25,
            "nfft": 64,
            "detrend": False,
            "return_onesided": False,
            "boundary": "even",
            "padded": False,
            "axis": 1,
        }
    )

    # Input 8: 1D array, minimal overlap (noverlap=1)
    nperseg = 32
    x = np.random.randn(150).astype(np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 50.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 1,
            "nfft": 64,
            "detrend": False,
            "return_onesided": True,
            "boundary": "constant",
            "padded": True,
            "axis": -1,
        }
    )

    # Input 9: 3D array computing along intermediate axis (axis=1)
    nperseg = 64
    x = np.random.randn(4, 256, 6).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 200.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 32,
            "nfft": 64,
            "detrend": False,
            "return_onesided": True,
            "boundary": "odd",
            "padded": True,
            "axis": 1,
        }
    )

    # Input 10: Large segment size with float64 precision
    nperseg = 256
    x = np.random.randn(1024).astype(np.float64)
    window = np.blackman(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 1000.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 128,
            "nfft": 512,
            "detrend": False,
            "return_onesided": True,
            "boundary": "zeros",
            "padded": False,
            "axis": -1,
        }
    )

    # Input 11: 2D array, negative axis indexing (axis=-2)
    nperseg = 16
    x = np.random.randn(64, 10).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "fs": 0.5,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 8,
            "nfft": 32,
            "detrend": False,
            "return_onesided": False,
            "boundary": "even",
            "padded": True,
            "axis": -2,
        }
    )

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.scipy.signal.stft_2"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_2'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_2'], lib="jax", suffix=2)
