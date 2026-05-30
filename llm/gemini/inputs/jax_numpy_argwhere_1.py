
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argwhere_inputs():
    list_of_inputs = []

    # Input 1, 2D array with 1s and 0s
    a = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    size = 5
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 2, 1D array with positive, negative, and zero values
    a = np.array([0, -1, 3, 0, 5, -2, 0], dtype=np.int32)
    size = 3
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 3, 3D binary array
    a = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=np.int32)
    size = 10
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 4, 1D boolean array
    a = np.array([True, False, True, False, True], dtype=bool)
    size = 4
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 5, 2D float array with zeros and non-zeros
    a = np.array([[0.0, 1.5, -2.5], [0.0, 0.0, 3.1]], dtype=np.float32)
    size = 8
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 6, 4D array of ones
    a = np.ones((2, 2, 2, 2), dtype=np.int32)
    size = 20
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 7, 2D array of all zeros
    a = np.array([[0, 0], [0, 0]], dtype=np.int32)
    size = 4
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 8, 1D array where size is smaller than the number of non-zero elements
    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    size = 2
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 9, 0D scalar array (nonzero)
    a = np.array(5, dtype=np.int32)
    size = 1
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    # Input 10, 2D array with range elements
    a = np.arange(-5, 5, dtype=np.int32).reshape(2, 5)
    size = 12
    fill_value = np.array(-2, dtype=np.int32)
    list_of_inputs.append({"a": a, "size": size, "fill_value": fill_value})

    return list_of_inputs

generated_inputs["jax.numpy.argwhere_1"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_1'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_1'], lib="jax", suffix=1)
