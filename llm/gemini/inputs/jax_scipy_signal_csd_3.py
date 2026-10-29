
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def csd_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D real arrays, mean averaging, density scaling, onesided
    nperseg = 64
    x = np.random.randn(256).astype(np.float32)
    y = np.random.randn(256).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 1.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 32,
            "nfft": 64,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 2: 1D real arrays with zero padding (nfft > nperseg), linear detrend, median averaging
    nperseg = 128
    x = np.random.randn(512).astype(np.float64)
    y = np.random.randn(512).astype(np.float64)
    window = np.hamming(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 100.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 64,
            "nfft": 256,
            "detrend": "linear",
            "return_onesided": False,
            "scaling": "spectrum",
            "axis": -1,
            "average": "median",
        }
    )

    # Input 3: 2D real arrays, axis=1, no overlap
    nperseg = 64
    x = np.random.randn(4, 256).astype(np.float32)
    y = np.random.randn(4, 256).astype(np.float32)
    window = np.blackman(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 250.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 0,
            "nfft": 64,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": 1,
            "average": "mean",
        }
    )

    # Input 4: 2D real arrays, axis=0
    nperseg = 128
    x = np.random.randn(512, 3).astype(np.float32)
    y = np.random.randn(512, 3).astype(np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 50.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 32,
            "nfft": 128,
            "detrend": "linear",
            "return_onesided": True,
            "scaling": "spectrum",
            "axis": 0,
            "average": "mean",
        }
    )

    # Input 5: 1D complex arrays, return_onesided=False
    nperseg = 64
    x = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
    y = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 10.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 16,
            "nfft": 64,
            "detrend": "constant",
            "return_onesided": False,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 6: 1D complex arrays with float64/complex128 precision, median averaging
    nperseg = 128
    x = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex128)
    y = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex128)
    window = np.hamming(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 0.5,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 64,
            "nfft": 128,
            "detrend": "linear",
            "return_onesided": False,
            "scaling": "spectrum",
            "axis": -1,
            "average": "median",
        }
    )

    # Input 7: 3D real arrays, axis=-1
    nperseg = 32
    x = np.random.randn(2, 3, 256).astype(np.float32)
    y = np.random.randn(2, 3, 256).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 2.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 16,
            "nfft": 64,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "median",
        }
    )

    # Input 8: 2D real arrays with float64, scaling='density', linear detrend
    nperseg = 50
    x = np.random.randn(10, 200).astype(np.float64)
    y = np.random.randn(10, 200).astype(np.float64)
    window = np.blackman(nperseg).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 20.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 25,
            "nfft": 50,
            "detrend": "linear",
            "return_onesided": False,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }
    )

    # Input 9: Short 1D real signals with small nperseg and padded nfft
    nperseg = 16
    x = np.random.randn(128).astype(np.float32)
    y = np.random.randn(128).astype(np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 8.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 8,
            "nfft": 32,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "spectrum",
            "axis": 0,
            "average": "mean",
        }
    )

    # Input 10: 3D complex arrays, axis=1
    nperseg = 64
    x = (np.random.randn(4, 128, 2) + 1j * np.random.randn(4, 128, 2)).astype(
        np.complex64
    )
    y = (np.random.randn(4, 128, 2) + 1j * np.random.randn(4, 128, 2)).astype(
        np.complex64
    )
    window = np.ones(nperseg, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "fs": 1000.0,
            "window": window,
            "nperseg": nperseg,
            "noverlap": 32,
            "nfft": 64,
            "detrend": "constant",
            "return_onesided": False,
            "scaling": "density",
            "axis": 1,
            "average": "mean",
        }
    )

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
