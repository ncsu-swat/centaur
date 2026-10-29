
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of same shape
    x = np.array([1, 5, 3, -2], dtype=np.int32)
    y = np.array([2, 4, 3, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D float32 arrays with negative and positive values
    x = np.array([-1.5, 0.0, 2.3, -10.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.1, -5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D array and 1D array with broadcasting
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    y = np.array([2, 2, 5], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar arrays
    x = np.array(3.5, dtype=np.float64)
    y = np.array(7.2, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 arrays of identical shapes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Multi-dimensional broadcasting with shapes (2, 1, 4) and (1, 3, 4)
    x = np.arange(8, dtype=np.int32).reshape(2, 1, 4)
    y = np.arange(12, dtype=np.int32).reshape(1, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Unsigned integer arrays (uint8)
    x = np.array([0, 128, 255], dtype=np.uint8)
    y = np.array([1, 100, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Float64 arrays containing infinity values
    x = np.array([-np.inf, 0.0, np.inf], dtype=np.float64)
    y = np.array([0.0, 0.0, np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D broadcastable arrays
    x = np.ones((1, 2, 1, 5), dtype=np.float32)
    y = np.zeros((3, 1, 4, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Column vector and row vector broadcasting (shape (3, 1) vs (1, 4))
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[0, 2, 4, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.less_1"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_1'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_1'], lib="jax", suffix=1)
