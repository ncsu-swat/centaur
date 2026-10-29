
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class WinTuple(tuple):

    def __len__(self):
        return 0


def csd_inputs():
    list_of_inputs = []

    # Input 1: 1D real signals, standard defaults, Hann window
    x = np.random.randn(256).astype(np.float32)
    y = np.random.randn(256).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 1.0,
        "window": WinTuple(("hann",)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D real signals, float64, Hamming window, spectrum scaling, linear detrend
    x = np.random.randn(512).astype(np.float64)
    y = np.random.randn(512).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 100.0,
        "window": WinTuple(("hamming",)),
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 0,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D real signals, axis=-1, boxcar window
    x = np.random.randn(4, 256).astype(np.float32)
    y = np.random.randn(4, 256).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 50.0,
        "window": WinTuple(("boxcar",)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D real signals, axis=0, Bartlett window
    x = np.random.randn(200, 8).astype(np.float32)
    y = np.random.randn(200, 8).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 10.0,
        "window": WinTuple(("bartlett",)),
        "nperseg": 50,
        "noverlap": 25,
        "nfft": 64,
        "detrend": "linear",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex signals, return_onesided=False, Blackman window
    x = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
    y = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 256.0,
        "window": WinTuple(("blackman",)),
        "nperseg": 64,
        "noverlap": 0,
        "nfft": 64,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D real signals, axis=1, triang window
    x = np.random.randn(2, 128, 3).astype(np.float32)
    y = np.random.randn(2, 128, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 1000.0,
        "window": WinTuple(("triang",)),
        "nperseg": 64,
        "noverlap": 16,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D real signals, zero overlap, bohman window
    x = np.random.randn(1024).astype(np.float64)
    y = np.random.randn(1024).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 0.5,
        "window": WinTuple(("bohman",)),
        "nperseg": 128,
        "noverlap": 0,
        "nfft": 256,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex signals, return_onesided=False, parzen window
    x = (np.random.randn(6, 256) + 1j * np.random.randn(6, 256)).astype(
        np.complex128
    )
    y = (np.random.randn(6, 256) + 1j * np.random.randn(6, 256)).astype(
        np.complex128
    )
    input_dict = {
        "x": x,
        "y": y,
        "fs": 20.0,
        "window": WinTuple(("parzen",)),
        "nperseg": 80,
        "noverlap": 40,
        "nfft": 80,
        "detrend": "constant",
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D real signals with negative values, nuttall window
    x = -np.abs(np.random.randn(128).astype(np.float32))
    y = -np.abs(np.random.randn(128).astype(np.float32))
    input_dict = {
        "x": x,
        "y": y,
        "fs": 2.5,
        "window": WinTuple(("nuttall",)),
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": "linear",
        "return_onesided": False,
        "scaling": "density",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D real signals, axis=-1, flattop window
    x = np.random.randn(2, 2, 2, 128).astype(np.float32)
    y = np.random.randn(2, 2, 2, 128).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "fs": 8000.0,
        "window": WinTuple(("flattop",)),
        "nperseg": 32,
        "noverlap": 8,
        "nfft": 128,
        "detrend": "constant",
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.csd_2"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_2'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_2'], lib="jax", suffix=2)
