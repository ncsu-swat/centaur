
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def csd_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32, default-like settings
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 1.0,
            "window": "hann",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 2: 1D float64, spectrum scaling, hamming window
    x = np.random.randn(64).astype(np.float64)
    y = np.random.randn(64).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 10.0,
            "window": "hamming",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "spectrum",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 3: 2D array along axis=-1, boxcar window
    x = np.random.randn(4, 64).astype(np.float32)
    y = np.random.randn(4, 64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 2.0,
            "window": "boxcar",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 4: 2D array along axis=0, two-sided spectrum
    x = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64, 4).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 5.0,
            "window": "hann",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": False,
            "scaling": "density",
            "axis": 0,
            "average": "mean",
        }
    )

    # Input 5: Zero overlap (noverlap=0)
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 1.0,
            "window": "hann",
            "nperseg": 16,
            "noverlap": 0,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 6: Zero-padded FFT (nfft > nperseg)
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 20.0,
            "window": "hamming",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 32,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 7: 3D array along axis=-1, spectrum scaling
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = np.random.randn(2, 4, 64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 50.0,
            "window": "boxcar",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "spectrum",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 8: Two-sided spectrum on 1D signal
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 8.0,
            "window": "hann",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": False,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 9: Median averaging
    x = np.random.randn(4, 64).astype(np.float32)
    y = np.random.randn(4, 64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 100.0,
            "window": "hamming",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "median",
        }
    )

    # Input 10: Linear detrending
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 4.0,
            "window": "hann",
            "nperseg": 16,
            "noverlap": 8,
            "nfft": 16,
            "detrend": "linear",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.signal.csd_1"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_1'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_1'], lib="jax", suffix=1)
