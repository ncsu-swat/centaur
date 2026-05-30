
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1, 1D array of integers, axis 0, keepdims False
    a = np.array([1, 0, -3, 0, 5], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, 2D array of floats, axis 1, keepdims True
    a = np.array([[0.0, 1.5, 0.0], [-2.3, 0.0, 4.1]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, 3D array of int64, axis 2, keepdims False
    a = np.random.choice([0, 1, 2], size=(2, 3, 4)).astype(np.int64)
    input_dict = {"a": a, "axis": 2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, 2D boolean array, axis 0, keepdims True
    a = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, 4D array of float64, axis 1, keepdims False
    a = np.random.choice([0.0, -1.0, 2.0], size=(2, 2, 2, 2)).astype(np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, 1D array of complex numbers, axis 0, keepdims True
    a = np.array([0+0j, 1+1j, -1-1j], dtype=np.complex64)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, 3D array of int16 with negative axis, keepdims False
    a = np.random.choice([-10, 0, 10], size=(3, 3, 3)).astype(np.int16)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, 2D array of uint8, negative axis, keepdims True
    a = np.array([[0, 2, 0], [4, 0, 6]], dtype=np.uint8)
    input_dict = {"a": a, "axis": -2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, 5D array of float32, axis 3, keepdims True
    a = np.random.choice([0.0, 1.0], size=(2, 1, 2, 3, 2)).astype(np.float32)
    input_dict = {"a": a, "axis": 3, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, 2D array of zeros, axis 1, keepdims False
    a = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.count_nonzero_1"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_1'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_1'], lib="jax", suffix=1)
