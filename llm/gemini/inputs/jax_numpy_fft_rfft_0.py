
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfft_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, norm="backward"
    a = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, norm="ortho"
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with padding (n > dimension), float32, norm="forward"
    a = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 6,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with truncation (n < dimension), float64, norm="backward"
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 3,
        "axis": 1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D integer array, norm="ortho"
    a = np.random.randint(-10, 10, size=(16,)).astype(np.int32)
    input_dict = {
        "a": a,
        "n": 16,
        "axis": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, float32, norm="forward", target axis = 2
    a = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array, norm="backward", axis = 0
    a = np.random.randint(-5, 5, size=(8, 8)).astype(np.int64)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, float32, negative axis, norm="ortho"
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": -2,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with negative values, float64, norm="forward"
    a = np.array([-1.2, -0.5, 3.4, -5.6, 7.8, -9.0, 2.1, 0.0]).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 12,
        "axis": 0,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D high-dimensional array, float32, norm="backward"
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 2,
        "axis": -4,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfft"] = rfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft'.")


check_valid('jax.numpy.fft.rfft', generated_inputs['jax.numpy.fft.rfft'], lib="jax", suffix=0)
