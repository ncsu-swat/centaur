
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def welch_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, default standard settings
    nperseg = 32
    x = np.random.randn(256).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 1.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 16,
        "nfft": 32,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, linear detrend, two-sided, spectrum scaling
    nperseg = 64
    x = np.random.randn(512).astype(np.float64)
    window = np.hamming(nperseg).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 10.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 32,
        "nfft": 128,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 0, median averaging
    nperseg = 50
    x = np.random.randn(200, 4).astype(np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    input_dict = {
        "x": x,
        "fs": 50.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 25,
        "nfft": 50,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 0,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array along axis 1, noverlap=0, zero-padded FFT
    nperseg = 16
    x = np.random.randn(8, 128).astype(np.float32)
    window = np.blackman(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 0.5,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 0,
        "nfft": 32,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array along axis -1, median averaging, spectrum scaling
    nperseg = 64
    x = np.random.randn(2, 3, 256).astype(np.float32)
    window = np.bartlett(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 100.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 16,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with high sampling frequency and large FFT
    nperseg = 128
    x = np.random.randn(1024).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 1000.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "density",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, small overlap, two-sided
    nperseg = 32
    x = np.random.randn(6, 128).astype(np.float64)
    window = np.hamming(nperseg).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 25.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 8,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array along middle axis (axis=1)
    nperseg = 20
    x = np.random.randn(4, 100, 5).astype(np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    input_dict = {
        "x": x,
        "fs": 2.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 10,
        "nfft": 40,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D short array, equal nperseg and nfft
    nperseg = 16
    x = np.random.randn(64).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 2.5,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 8,
        "nfft": 16,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array along axis 2
    nperseg = 16
    x = np.random.randn(2, 2, 64, 3).astype(np.float32)
    window = np.blackman(nperseg).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 8.0,
        "window": window,
        "nperseg": nperseg,
        "noverlap": 4,
        "nfft": 32,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 2,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.welch_3"] = welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_3'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_3'], lib="jax", suffix=3)
