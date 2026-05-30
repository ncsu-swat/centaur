
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 1D int32 arrays with negative values
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Boolean arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 2D int32 arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting (1D to 2D) int32
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Broadcasting (scalar-like array to 3D) int32
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([1], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Large integers (int64) 1D
    x = np.array([12345678901234, 98765432109876], dtype=np.int64)
    y = np.array([98765432109876, 12345678901234], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Boolean arrays with broadcasting (2D and 1D)
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([True, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int32)
    y = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 4D int64 arrays
    x = np.random.randint(-1000, 1000, size=(2, 2, 2, 2)).astype(np.int64)
    y = np.random.randint(-1000, 1000, size=(2, 2, 2, 2)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 2D boolean arrays
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([[False, True], [True, False]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_and'.")


check_valid('jax.numpy.bitwise_and', generated_inputs['jax.numpy.bitwise_and'], lib="jax", suffix=0)
