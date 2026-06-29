
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hfft_inputs():
    list_of_inputs = []

    # Input 1: 1D real array, default-like parameters
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 18,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex array with forward normalization
    a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 14,
        "axis": 0,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D real array, float64, ortho normalization
    a = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex array, specifying axis 0
    a = (np.random.randn(6, 4) + 1j * np.random.randn(6, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "n": 10,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex array, middle axis
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D real array, negative axis
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 6,
        "axis": -2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D real array with odd n (produces odd-length output)
    a = np.random.randn(6).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 11,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D real array
    a = np.random.randn(2, 2, 3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 9,
        "axis": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex array, odd output length
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 7,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array, larger size
    a = np.random.randn(12).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 22,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.hfft"] = hfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.hfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.hfft'.")


check_valid('jax.numpy.fft.hfft', generated_inputs['jax.numpy.fft.hfft'], lib="jax", suffix=0)
