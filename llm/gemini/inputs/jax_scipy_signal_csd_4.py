
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def csd_inputs():
    list_of_inputs = []

    # Input 1: 1D signals, default-like parameters
    x = np.random.randn(512).astype(np.float32)
    y = np.random.randn(512).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 1.0,
        "window": "hann",
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D signals, float64, zero-padding with nfft > nperseg, linear detrend, two-sided
    x = np.random.randn(1024).astype(np.float64)
    y = np.random.randn(1024).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 100.0,
        "window": "hamming",
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 512,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D signals, along axis 1, boxcar window
    x = np.random.randn(4, 256).astype(np.float32)
    y = np.random.randn(4, 256).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 250.5,
        "window": "boxcar",
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D signals, computed along axis 0
    x = np.random.randn(200, 8).astype(np.float32)
    y = np.random.randn(200, 8).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 50.0,
        "window": "bartlett",
        "nperseg": 50,
        "noverlap": 25,
        "nfft": 100,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex inputs, two-sided spectrum
    x = (np.random.randn(500) + 1j * np.random.randn(500)).astype(np.complex64)
    y = (np.random.randn(500) + 1j * np.random.randn(500)).astype(np.complex64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 10.0,
        "window": "hann",
        "nperseg": 100,
        "noverlap": 50,
        "nfft": 100,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D signals, blackman window, median averaging
    x = np.random.randn(2, 3, 256).astype(np.float32)
    y = np.random.randn(2, 3, 256).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 2.0,
        "window": "blackman",
        "nperseg": 128,
        "noverlap": 0,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 signals, non-standard overlap
    x = np.random.randn(16, 512).astype(np.float64)
    y = np.random.randn(16, 512).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 1000.0,
        "window": "hann",
        "nperseg": 256,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D signals with negative values, low sampling frequency
    x = np.random.uniform(-10.0, -1.0, size=(300,)).astype(np.float32)
    y = np.random.uniform(-5.0, 5.0, size=(300,)).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 0.5,
        "window": "hamming",
        "nperseg": 60,
        "noverlap": 30,
        "nfft": 120,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": 0,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D signals along axis 0
    x = np.random.randn(100, 4, 8).astype(np.float32)
    y = np.random.randn(100, 4, 8).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 12.5,
        "window": "boxcar",
        "nperseg": 50,
        "noverlap": 20,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Short signals with bartlett window
    x = np.random.randn(128).astype(np.float32)
    y = np.random.randn(128).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 8.0,
        "window": "bartlett",
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 32,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.csd_4"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_4'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_4'], lib="jax", suffix=4)
