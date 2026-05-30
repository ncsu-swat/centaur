
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def remainder_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays of int32 (positive values)
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([3, 4, 5, 6], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D arrays with negative dividend (x1)
    x1 = np.array([-10, -20, -30, -40], dtype=np.int32)
    x2 = np.array([3, 4, 5, 6], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 1D arrays with negative divisor (x2)
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([-3, -4, -5, -6], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 2D arrays of float32, same shape
    x1 = np.array([[5.5, 8.2], [9.1, 11.7]], dtype=np.float32)
    x2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Broadcasting - 2D and 1D arrays (float32)
    x1 = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Broadcasting - different shapes (2, 1) and (1, 3)
    x1 = np.array([[10], [20]], dtype=np.int32)
    x2 = np.array([[3, 4, 5]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 3D arrays of float64, same shape
    x1 = np.random.uniform(10.0, 50.0, size=(2, 2, 2)).astype(np.float64)
    x2 = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 1D array and 0D array (scalar equivalent)
    x1 = np.array([15, 25, 35], dtype=np.int64)
    x2 = np.array(4, dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Large integer values (int64)
    x1 = np.array([100000000000, -200000000000], dtype=np.int64)
    x2 = np.array([3000000000, 7000000000], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 4D arrays of float32
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(1.0, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.remainder_1"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_1'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_1'], lib="jax", suffix=1)
