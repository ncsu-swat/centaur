
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, size < actual non-zeros (truncation)
    a = np.array([1, 0, -2, 0, 3, -4, 0], dtype=np.int32)
    size = 2
    fill_value = (np.array(-1, dtype=np.int32),)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 2: 2D float32 array, size > actual non-zeros (padding)
    a = np.array([[0.0, 1.5], [2.3, 0.0]], dtype=np.float32)
    size = 5
    fill_value = (np.array(-1, dtype=np.int32), np.array(-1, dtype=np.int32))
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 3: 3D int64 array
    a = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=np.int64)
    size = 6
    fill_value = (np.array(0, dtype=np.int32), np.array(0, dtype=np.int32), np.array(0, dtype=np.int32))
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 4: 1D float64 array
    a = np.array([0.0, -1.1, 0.0, 2.2, 3.3], dtype=np.float64)
    size = 3
    fill_value = (np.array(99, dtype=np.int32),)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 5: 2D int32 array, completely filled with non-zeros
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    size = 8
    fill_value = (np.array(-9, dtype=np.int32), np.array(-9, dtype=np.int32))
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 6: 4D float32 array with sparse non-zeros
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    a[0, 1, 0, 1] = 4.2
    a[1, 0, 1, 0] = -1.7
    size = 3
    fill_value = (
        np.array(-1, dtype=np.int32),
        np.array(-1, dtype=np.int32),
        np.array(-1, dtype=np.int32),
        np.array(-1, dtype=np.int32)
    )
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 7: 1D int32 array, all zeros
    a = np.zeros((10,), dtype=np.int32)
    size = 3
    fill_value = (np.array(-1, dtype=np.int32),)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 8: 2D int64 array
    a = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.int64)
    size = 5
    fill_value = (np.array(999, dtype=np.int32), np.array(999, dtype=np.int32))
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 9: 3D float64 array
    a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]], dtype=np.float64)
    size = 2
    fill_value = (np.array(-1, dtype=np.int32), np.array(-1, dtype=np.int32), np.array(-1, dtype=np.int32))
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 10: 1D int32 array, size exactly matching non-zeros
    a = np.array([10, 20, 0, 30, 40], dtype=np.int32)
    size = 4
    fill_value = (np.array(5, dtype=np.int32),)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    return list_of_inputs

generated_inputs["jax.numpy.nonzero_2"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nonzero_2'.")


check_valid('jax.numpy.nonzero', generated_inputs['jax.numpy.nonzero_2'], lib="jax", suffix=2)
