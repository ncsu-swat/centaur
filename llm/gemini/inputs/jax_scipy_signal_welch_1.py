
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def welch_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like parameters
    x = np.random.randn(1024).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 1.0,
        "window": "hann",
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with higher sampling rate and spectrum scaling
    x = np.random.randn(2048).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 1000.0,
        "window": "hamming",
        "nperseg": 512,
        "noverlap": 256,
        "nfft": 512,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, computed along axis 0
    x = np.random.randn(500, 10).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 50.0,
        "window": "boxcar",
        "nperseg": 100,
        "noverlap": 50,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, computed along axis 1 (default last axis)
    x = np.random.randn(8, 1000).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 256.0,
        "window": "blackman",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=2
    x = np.random.randn(4, 5, 600).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 100.0,
        "window": "bartlett",
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 2,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, negative axis index (axis=-2)
    x = np.random.randn(4, 800, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 10.0,
        "window": "hann",
        "nperseg": 200,
        "noverlap": 100,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": -2,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero overlap
    x = np.random.randn(1000).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 0.5,
        "window": "hann",
        "nperseg": 100,
        "noverlap": 0,
        "nfft": 100,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "density",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High overlap with nfft > nperseg (zero-padding FFT)
    x = np.random.randn(1500).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 2000.0,
        "window": "hamming",
        "nperseg": 256,
        "noverlap": 200,
        "nfft": 512,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small segment size and 4D array
    x = np.random.randn(2, 3, 4, 300).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 128.0,
        "window": "boxcar",
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-power-of-two nperseg and nfft
    x = np.random.randn(750).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 44100.0,
        "window": "blackman",
        "nperseg": 150,
        "noverlap": 75,
        "nfft": 300,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.welch_1"] = welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_1'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_1'], lib="jax", suffix=1)
