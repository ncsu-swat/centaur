
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float arrays with same shape
    x = np.array([-4.0, 1.5, 7.2], dtype=np.float32)
    y = np.array([2.0, -3.1, 7.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D integer arrays with negative values
    x = np.array([-10, 0, 5, -3], dtype=np.int32)
    y = np.array([-5, 0, 4, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D float64 arrays
    x = np.array([[-1.0, 2.5], [3.0, -4.2]], dtype=np.float64)
    y = np.array([[0.0, 2.5], [1.5, -5.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting 1D to 2D
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([[0, 2, 5], [2, 2, 2]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcasting with a singleton dimension
    x = np.array([[1.0], [4.0], [-2.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0, -2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalar arrays
    x = np.array(5, dtype=np.int32)
    y = np.array(-2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 3D float arrays
    x = np.zeros((2, 3, 4), dtype=np.float32)
    y = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Unsigned integer arrays
    x = np.array([0, 15, 255], dtype=np.uint8)
    y = np.array([1, 15, 100], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D arrays with broadcasting
    x = np.random.randn(2, 1, 4, 1).astype(np.float32)
    y = np.random.randn(1, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Arrays with special float values (inf, nan)
    x = np.array([-np.inf, 0.0, np.inf, np.nan], dtype=np.float32)
    y = np.array([0.0, 0.0, np.inf, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

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
