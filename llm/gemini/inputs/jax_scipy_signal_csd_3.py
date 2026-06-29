
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def csd_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "x": np.random.randn(128).astype(np.float32),
        "y": np.random.randn(128).astype(np.float32),
        "fs": 100.0,
        "window": np.hanning(32).astype(np.float32),
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "x": np.random.randn(256).astype(np.float64),
        "y": np.random.randn(256).astype(np.float64),
        "fs": 1.0,
        "window": np.hamming(64).astype(np.float64),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": 0,
        "average": "median"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "x": np.random.randn(10, 500).astype(np.float32),
        "y": np.random.randn(10, 500).astype(np.float32),
        "fs": 200.5,
        "window": np.ones(128).astype(np.float32),
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "x": np.random.randn(500, 10).astype(np.float32),
        "y": np.random.randn(500, 10).astype(np.float32),
        "fs": 50.0,
        "window": np.hanning(100).astype(np.float32),
        "nperseg": 100,
        "noverlap": 50,
        "nfft": 128,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "density",
        "axis": 0,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "x": (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64),
        "y": (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64),
        "fs": 1000.0,
        "window": np.blackman(200).astype(np.float32),
        "nperseg": 200,
        "noverlap": 100,
        "nfft": 200,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "x": np.random.randn(64).astype(np.float32),
        "y": np.random.randn(64).astype(np.float32),
        "fs": 8.0,
        "window": np.hamming(8).astype(np.float32),
        "nperseg": 8,
        "noverlap": 4,
        "nfft": 8,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "x": np.random.randn(4, 4, 128).astype(np.float32),
        "y": np.random.randn(4, 4, 128).astype(np.float32),
        "fs": 12.5,
        "window": np.hanning(32).astype(np.float32),
        "nperseg": 32,
        "noverlap": 0,
        "nfft": 32,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "x": np.random.randn(128, 4, 4).astype(np.float32),
        "y": np.random.randn(128, 4, 4).astype(np.float32),
        "fs": 1.0,
        "window": np.hamming(16).astype(np.float32),
        "nperseg": 16,
        "noverlap": 8,
        "nfft": 32,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "density",
        "axis": 0,
        "average": "median"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "x": np.random.randn(16).astype(np.float32),
        "y": np.random.randn(16).astype(np.float32),
        "fs": 2.0,
        "window": np.ones(4).astype(np.float32),
        "nperseg": 4,
        "noverlap": 0,
        "nfft": 4,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "x": np.random.randn(50, 50).astype(np.float64),
        "y": np.random.randn(50, 50).astype(np.float64),
        "fs": 44100.0,
        "window": np.blackman(10).astype(np.float64),
        "nperseg": 10,
        "noverlap": 5,
        "nfft": 16,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 1,
        "average": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.csd_3"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_3'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_3'], lib="jax", suffix=3)
