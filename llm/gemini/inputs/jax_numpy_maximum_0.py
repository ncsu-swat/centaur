
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def maximum_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of same shape with positive and negative integers
    x = np.array([1, -5, 3, 2], dtype=np.int32)
    y = np.array([-2, 4, 7, -6], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D float32 arrays with broadcasting (2, 3) and (1, 3)
    x = np.array([[-2.5, 5.0, 7.2], [1.1, -6.3, 3.0]], dtype=np.float32)
    y = np.array([[-5.0, 3.0, 9.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 3D float64 arrays of same shape
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 0D scalar arrays
    x = np.array(3.14, dtype=np.float32)
    y = np.array(-2.71, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Arrays with NaN and Inf values
    x = np.array([np.nan, -np.inf, np.inf, 2.0], dtype=np.float32)
    y = np.array([1.0, 0.0, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: int64 arrays
    x = np.array([[-100, 200], [300, -400]], dtype=np.int64)
    y = np.array([[50, -250], [-350, 450]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: uint8 arrays
    x = np.array([0, 128, 255], dtype=np.uint8)
    y = np.array([10, 100, 200], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: boolean arrays
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 4D arrays with multi-axis broadcasting
    x = np.random.randn(2, 1, 4, 1).astype(np.float32)
    y = np.random.randn(1, 3, 1, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 1D and 2D broadcasting
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array([[0.0], [2.0], [-1.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Empty array with zero dimension
    x = np.empty((0, 4), dtype=np.float32)
    y = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs


generated_inputs["jax.numpy.maximum"] = maximum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.maximum'.")


check_valid('jax.numpy.maximum', generated_inputs['jax.numpy.maximum'], lib="jax", suffix=0)
