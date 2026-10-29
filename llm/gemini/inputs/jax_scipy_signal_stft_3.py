
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class WindowTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array([0.0])

def stft_inputs():
    list_of_inputs = []

    # Input 1: 1D real signal, Hann window
    x = np.random.randn(1024).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 1.0,
        "window": WindowTuple(("hann",)),
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

    # Input 2: 1D float32 signal with boxcar window and two-sided spectrum
    x = np.random.randn(500).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 100.0,
        "window": WindowTuple(("boxcar",)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "boundary": "even",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis -1, Hamming window
    x = np.random.randn(10, 512).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 16000.0,
        "window": WindowTuple(("hamming",)),
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 128,
        "detrend": False,
        "return_onesided": True,
        "boundary": "odd",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array along axis 0, Blackman window, no overlap
    x = np.random.randn(512, 8).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 8000.0,
        "window": WindowTuple(("blackman",)),
        "nperseg": 64,
        "noverlap": 0,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "boundary": "constant",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex 1D signal with Kaiser window
    x = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
    input_dict = {
        "x": x,
        "fs": 44100.0,
        "window": WindowTuple(("kaiser", 4.0)),
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 512,
        "detrend": False,
        "return_onesided": False,
        "boundary": "zeros",
        "padded": True,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 tensor with Tukey window
    x = np.random.randn(4, 8, 256).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 250.0,
        "window": WindowTuple(("tukey", 0.25)),
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": False,
        "return_onesided": False,
        "boundary": "zeros",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 tensor along axis 0 with Bartlett window
    x = np.random.randn(128, 4, 16).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 500.0,
        "window": WindowTuple(("bartlett",)),
        "nperseg": 64,
        "noverlap": 48,
        "nfft": 128,
        "detrend": False,
        "return_onesided": True,
        "boundary": "even",
        "padded": True,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D signal with Gaussian window
    x = np.random.randn(2048).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 2.0,
        "window": WindowTuple(("gaussian", 2.0)),
        "nperseg": 512,
        "noverlap": 256,
        "nfft": 512,
        "detrend": False,
        "return_onesided": True,
        "boundary": "constant",
        "padded": False,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex128 array along axis 1 with cosine window
    x = (np.random.randn(16, 256) + 1j * np.random.randn(16, 256)).astype(np.complex128)
    input_dict = {
        "x": x,
        "fs": 1000.0,
        "window": WindowTuple(("cosine",)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": False,
        "return_onesided": False,
        "boundary": "odd",
        "padded": True,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array along middle axis 1 with flattop window
    x = np.random.randn(2, 500, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "fs": 12.5,
        "window": WindowTuple(("flattop",)),
        "nperseg": 100,
        "noverlap": 50,
        "nfft": 100,
        "detrend": False,
        "return_onesided": False,
        "boundary": "zeros",
        "padded": True,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small 1D signal with parzen window
    x = np.random.randn(128).astype(np.float64)
    input_dict = {
        "x": x,
        "fs": 0.5,
        "window": WindowTuple(("parzen",)),
        "nperseg": 16,
        "noverlap": 8,
        "nfft": 32,
        "detrend": False,
        "return_onesided": True,
        "boundary": "even",
        "padded": False,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.stft_3"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_3'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_3'], lib="jax", suffix=3)
