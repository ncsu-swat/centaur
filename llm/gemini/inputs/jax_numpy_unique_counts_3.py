
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int array, size larger than unique count to trigger fill_value
    x = np.array([3, 1, 2, 1, 3], dtype=np.int32)
    size = 5
    fill_value = -1.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 2: 2D float array, negative values, size matches unique count
    x = np.array([[-1.5, 2.3], [-1.5, 0.0]], dtype=np.float32)
    size = 3
    fill_value = -9.9
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 3: 3D int array with lots of duplicate values, size smaller than unique count
    x = np.array([[[1, 2], [2, 3]], [[3, 4], [4, 5]]], dtype=np.int64)
    size = 3
    fill_value = 0.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 4: Large 1D random integers
    x = np.random.randint(-100, 100, size=(50,)).astype(np.int32)
    size = 10
    fill_value = 999.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 5: Float64 array with precise decimals, size requiring padding
    x = np.array([0.123456, -0.123456, 0.123456, 0.0], dtype=np.float64)
    size = 5
    fill_value = -1.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 6: 1D array of zeros and ones, size larger than unique count
    x = np.array([0, 1, 0, 1, 1, 0, 0], dtype=np.int32)
    size = 4
    fill_value = -99.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 7: Multi-dimensional array with identical values
    x = np.ones((3, 3), dtype=np.int32)
    size = 2
    fill_value = -5.5
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 8: 1D array of negative floats, size matches unique count
    x = np.array([-10.5, -20.5, -10.5, -30.5], dtype=np.float32)
    size = 3
    fill_value = 0.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 9: Highly redundant integers, size larger than unique count
    x = np.array([42] * 20, dtype=np.int32)
    size = 5
    fill_value = -42.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 10: 2D array of random floats
    x = np.random.uniform(-5.0, 5.0, size=(5, 5)).astype(np.float32)
    size = 15
    fill_value = -100.0
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    return list_of_inputs

generated_inputs["jax.numpy.unique_counts_3"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_3'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_3'], lib="jax", suffix=3)
