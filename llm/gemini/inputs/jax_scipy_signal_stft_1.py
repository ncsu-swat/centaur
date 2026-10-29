
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def stft_inputs():
    list_of_inputs = []

    # Input 1: 1D real float32 array, default-like parameters
    x = np.random.randn(1024).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 1.0,
        "window": "hann",
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 256,
        "detrend": False,
        "return_onesided": True,
        "boundary": "zeros",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D real float64 array with audio-like sampling frequency and hamming window
    x = np.random.randn(2048).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 16000.0,
        "window": "hamming",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 128,
        "detrend": False,
        "return_onesided": True,
        "boundary": "even",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D real float32 array (batch of signals), axis=-1
    x = np.random.randn(4, 512).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 100.0,
        "window": "boxcar",
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "boundary": "constant",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D real float32 array, transforming along axis 0
    x = np.random.randn(500, 8).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 50.0,
        "window": "blackman",
        "nperseg": 64,
        "noverlap": 16,
        "nfft": 128,
        "detrend": False,
        "return_onesided": True,
        "boundary": "zeros",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D real float32 array with two-sided spectrum (return_onesided=False)
    x = np.random.randn(800).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 256.0,
        "window": "bartlett",
        "nperseg": 100,
        "noverlap": 50,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "boundary": "odd",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array (must use return_onesided=False)
    x = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    input_dict = {
        "x": x,
        "fs": 44100.0,
        "window": "hann",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "boundary": "zeros",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D real float64 array with boundary='zeros', zero-padded FFT (nfft > nperseg)
    x = np.random.randn(2, 3, 400).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 2.5,
        "window": "hann",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": False,
        "return_onesided": True,
        "boundary": "zeros",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D real float32 array with zero overlap (noverlap=0)
    x = np.random.randn(512).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 8.0,
        "window": "boxcar",
        "nperseg": 64,
        "noverlap": 0,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "boundary": "zeros",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex128 array with odd boundary extension
    x = (np.random.randn(5, 600) + 1j * np.random.randn(5, 600)).astype(
        np.complex128
    )
    input_dict = {
        "x": x,
        "fs": 1000.0,
        "window": "hamming",
        "nperseg": 64,
        "noverlap": 48,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "boundary": "odd",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D real float32 array with negative axis and large overlap
    x = np.random.randn(300).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 500.0,
        "window": "blackman",
        "nperseg": 50,
        "noverlap": 40,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "boundary": "even",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.stft_1"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_1'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_1'], lib="jax", suffix=1)
