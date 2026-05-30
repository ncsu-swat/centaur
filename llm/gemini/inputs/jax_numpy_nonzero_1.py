
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nonzero_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    a = np.array([0, 1, 0, 3, 0], dtype=np.int32)
    size = 3
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 2: 2D array with some zeros
    a = np.array([[1, 0], [0, 2]], dtype=np.int32)
    size = 4
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 3: 3D array
    a = np.array([[[1, 0], [0, 0]], [[0, 2], [3, 4]]], dtype=np.int32)
    size = 2
    fill_value = np.array(99, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 4: 1D array with floats and negative values
    a = np.array([0.0, -1.5, 2.3, 0.0, -0.5], dtype=np.float32)
    size = 5
    fill_value = np.array(-9.0, dtype=np.float32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 5: 2D boolean array
    a = np.array([[True, False], [False, True]], dtype=bool)
    size = 1
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 6: 1D int64 array with larger size
    a = np.array([10, 20, 30, 0, 50, 0], dtype=np.int64)
    size = 10
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 7: 4D array with scattered non-zeros
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    a[0, 1, 0, 1] = 5
    a[1, 1, 1, 1] = 10
    size = 5
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 8: 1D array with all zeros
    a = np.zeros(10, dtype=np.int32)
    size = 3
    fill_value = np.array(9, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 9: 2D array with all non-zeros (size is smaller than count)
    a = np.ones((3, 3), dtype=np.int32)
    size = 5
    fill_value = np.array(-2, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 10: 1D array with negative elements
    a = np.array([-1, 0, -2, 0, -3], dtype=np.int32)
    size = 2
    fill_value = np.array(-5, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    return list_of_inputs

generated_inputs["jax.numpy.nonzero_1"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nonzero_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nonzero_1'.")


check_valid('jax.numpy.nonzero', generated_inputs['jax.numpy.nonzero_1'], lib="jax", suffix=1)
