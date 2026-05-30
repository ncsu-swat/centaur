
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def less_equal_inputs():
    list_of_inputs = []

    # Input 1: Small 1D arrays, float32, with positive and negative values
    x = np.array([-1.5, 0.0, 2.3], dtype=np.float32)
    y = np.array([-2.0, 0.0, 3.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D arrays of the same shape, int32
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 2], [2, 5]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D arrays of the same shape, float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcast compatible: 1D array (3,) and 2D array (2, 3)
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([[2, 2, 2], [0, 5, 1]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Broadcast compatible: 2D column vector (3, 1) and 2D row vector (1, 4)
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[2, 0, 4, 1]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Scalar-like arrays (0-dimensional)
    x = np.array(5, dtype=np.int32)
    y = np.array(10, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Unsigned integer arrays (uint8)
    x = np.array([0, 255, 128], dtype=np.uint8)
    y = np.array([1, 254, 128], dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large 4D arrays of same shape, float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Broadcast compatible: 0D array and 3D array
    x = np.array(1.5, dtype=np.float64)
    y = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.less_equal"] = less_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_equal'.")


check_valid('jax.numpy.less_equal', generated_inputs['jax.numpy.less_equal'], lib="jax", suffix=0)
