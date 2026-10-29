
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class WindowParam:
    def __init__(self, val=1.0):
        self.val = float(val)

    def __lt__(self, other):
        if isinstance(other, str):
            return False
        return self.val < getattr(other, "val", other)

    def __gt__(self, other):
        if isinstance(other, str):
            return True
        return self.val > getattr(other, "val", other)

    def __le__(self, other):
        if isinstance(other, str):
            return False
        return self.val <= getattr(other, "val", other)

    def __ge__(self, other):
        if isinstance(other, str):
            return True
        return self.val >= getattr(other, "val", other)

    def __float__(self):
        return self.val

    def __int__(self):
        return int(self.val)

def welch_inputs():
    list_of_inputs = []

    # Input 1: 1D real float32 array, default parameters
    x1 = np.random.randn(256).astype(np.float32)
    inp1 = {
        "x": x1,
        "fs": 1.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp1))

    # Input 2: 1D real float64 array, median averaging, two-sided spectrum
    x2 = np.random.randn(512).astype(np.float64)
    inp2 = {
        "x": x2,
        "fs": 2.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 128,
        "noverlap": 64,
        "nfft": 256,
        "detrend": False,
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": 0,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(inp2))

    # Input 3: 2D real float32 array, zero overlap
    x3 = np.random.randn(4, 512).astype(np.float32)
    inp3 = {
        "x": x3,
        "fs": 100.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 64,
        "noverlap": 0,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp3))

    # Input 4: 2D real float32 array, along axis 0
    x4 = np.random.randn(512, 4).astype(np.float32)
    inp4 = {
        "x": x4,
        "fs": 250.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 128,
        "noverlap": 32,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": 0,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp4))

    # Input 5: 3D real float64 array, median averaging
    x5 = np.random.randn(2, 8, 256).astype(np.float64)
    inp5 = {
        "x": x5,
        "fs": 50.5,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "scaling": "density",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(inp5))

    # Input 6: 1D complex64 array, two-sided spectrum
    x6 = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
    inp6 = {
        "x": x6,
        "fs": 10.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 64,
        "noverlap": 16,
        "nfft": 64,
        "detrend": False,
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp6))

    # Input 7: 2D complex128 array, axis 1, median averaging
    x7 = (np.random.randn(16, 256) + 1j * np.random.randn(16, 256)).astype(np.complex128)
    inp7 = {
        "x": x7,
        "fs": 1.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 64,
        "noverlap": 32,
        "nfft": 128,
        "detrend": False,
        "return_onesided": False,
        "scaling": "spectrum",
        "axis": 1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(inp7))

    # Input 8: 3D float32 array, along middle axis (axis 1)
    x8 = np.random.randn(4, 128, 8).astype(np.float32)
    inp8 = {
        "x": x8,
        "fs": 0.5,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 32,
        "noverlap": 8,
        "nfft": 64,
        "detrend": False,
        "return_onesided": True,
        "scaling": "density",
        "axis": 1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp8))

    # Input 9: 1D float32 array, high sampling rate
    x9 = np.random.randn(1024).astype(np.float32)
    inp9 = {
        "x": x9,
        "fs": 1000.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 256,
        "noverlap": 128,
        "nfft": 512,
        "detrend": False,
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "mean",
    }
    list_of_inputs.append(copy.deepcopy(inp9))

    # Input 10: 2D float64 array, non-power-of-two segment size
    x10 = np.random.randn(3, 300).astype(np.float64)
    inp10 = {
        "x": x10,
        "fs": 12.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 50,
        "noverlap": 25,
        "nfft": 100,
        "detrend": False,
        "return_onesided": False,
        "scaling": "density",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(inp10))

    # Input 11: 4D float32 array, negative axis indexing
    x11 = np.random.randn(2, 2, 4, 128).astype(np.float32)
    inp11 = {
        "x": x11,
        "fs": 4.0,
        "window": ("tukey", WindowParam(1.0)),
        "nperseg": 32,
        "noverlap": 16,
        "nfft": 32,
        "detrend": False,
        "return_onesided": True,
        "scaling": "spectrum",
        "axis": -1,
        "average": "median",
    }
    list_of_inputs.append(copy.deepcopy(inp11))

    return list_of_inputs

generated_inputs["jax.scipy.signal.welch_2"] = welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_2'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_2'], lib="jax", suffix=2)
