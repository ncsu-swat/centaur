
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def stft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard hann window
    x = np.random.randn(512).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 1.0,
        "window": "hann",
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "boundary": "zeros",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, hamming window, nfft > nperseg, detrend linear
    x = np.random.randn(1000).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 100.0,
        "window": "hamming",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "linear",
        "return_onesided": True,
        "boundary": "even",
        "padded": False,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, boxcar window, odd boundary
    x = np.random.randn(4, 1024).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 22050.0,
        "window": "boxcar",
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": True,
        "boundary": "odd",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array along axis 0, two-sided spectrum, bartlett window
    x = np.random.randn(256, 8).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 44100.0,
        "window": "bartlett",
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": "linear",
        "return_onesided": False,
        "boundary": "constant",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, blackman window, axis 1
    x = np.random.randn(3, 200, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 10.0,
        "window": "blackman",
        "nperseg": 16,
        "noverlap": 8,
        "nfft": 32,
        "detrend": "constant",
        "return_onesided": True,
        "boundary": "zeros",
        "padded": False,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array, return_onesided=False, zero overlap
    x = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
    input_dict = {
        "x": x,
        "fs": 2.0,
        "window": "hann",
        "nperseg": 64,
        "noverlap": 0,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": False,
        "boundary": "zeros",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, negative axis index -2
    x = np.random.randn(300, 16).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 0.5,
        "window": "hamming",
        "nperseg": 50,
        "noverlap": 25,
        "nfft": 100,
        "detrend": "linear",
        "return_onesided": False,
        "boundary": "even",
        "padded": True,
        "axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, high overlap ratio (75%), odd boundary
    x = np.random.randn(1024).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 8000.0,
        "window": "hann",
        "nperseg": 128,
        "noverlap": 96,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "boundary": "odd",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, axis 2, constant boundary
    x = np.random.randn(2, 4, 512).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 1000.0,
        "window": "blackman",
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 128,
        "detrend": "linear",
        "return_onesided": False,
        "boundary": "constant",
        "padded": True,
        "axis": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array, noverlap=0, small nperseg
    x = np.random.randn(128).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 50.0,
        "window": "boxcar",
        "nperseg": 32,
        "noverlap": 0,
        "nfft": 32,
        "detrend": "constant",
        "return_onesided": True,
        "boundary": "zeros",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.stft_4"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_4'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_4'], lib="jax", suffix=4)
