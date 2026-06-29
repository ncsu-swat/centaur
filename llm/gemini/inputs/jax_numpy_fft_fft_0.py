
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, backward norm
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64 array, ortho norm
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, forward norm
    a = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 16,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 array (with negative values), backward norm
    a = np.random.randint(-10, 10, size=(12,)).astype(np.int32)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, ortho norm
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 10,
        "axis": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D complex128 array, forward norm
    a = (np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)).astype(np.complex128)
    input_dict = {
        "a": a,
        "n": 3,
        "axis": 2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D strictly negative float32 array, backward norm
    a = np.random.uniform(-5.0, -1.0, size=(15,)).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 12,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int64 array, ortho norm
    a = np.random.randint(-100, 100, size=(4, 4, 4)).astype(np.int64)
    input_dict = {
        "a": a,
        "n": 2,
        "axis": 2,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array, backward norm
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 6,
        "axis": 1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, forward norm
    a = np.random.randn(2, 2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 3,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fft"] = fft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft'.")


check_valid('jax.numpy.fft.fft', generated_inputs['jax.numpy.fft.fft'], lib="jax", suffix=0)
