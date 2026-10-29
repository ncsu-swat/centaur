
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def minimum_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with matching shapes and mixed signs
    x = np.array([2, -3, 5, 1], dtype=np.int32)
    y = np.array([-3, 6, -4, 7], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D float32 arrays with broadcast compatibility
    x = np.array([[1.0, 5.0, 2.0], [-3.0, 4.0, 7.0]], dtype=np.float32)
    y = np.array([-2.0, 3.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 2D float32 arrays containing NaN and inf values
    x = np.array([[2.5, np.nan, -np.inf], [np.nan, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0, 5.0], [np.inf, np.nan, -1.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 0D scalar arrays
    x = np.array(-5.5, dtype=np.float64)
    y = np.array(3.2, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 3D float32 arrays with broadcasting across multiple dimensions
    x = np.random.randn(2, 1, 4).astype(np.float32)
    y = np.random.randn(1, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 1D uint8 arrays
    x = np.array([0, 128, 255, 10], dtype=np.uint8)
    y = np.array([1, 100, 250, 10], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 2D int64 arrays
    x = np.array([[-100, 200], [300, -400]], dtype=np.int64)
    y = np.array([[50, -50], [0, -500]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 4D float32 arrays of identical shapes
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4, 5)).astype(np.float32)
    y = np.random.uniform(-10.0, 10.0, size=(2, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Boolean arrays
    x = np.array([[True, False], [True, False]], dtype=np.bool_)
    y = np.array([[False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 1D float16 arrays
    x = np.array([-1.5, 0.0, 2.75, -8.125], dtype=np.float16)
    y = np.array([0.5, -0.0, 2.75, 4.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 3D array broadcast with a 1D array
    x = np.random.randn(4, 3, 2).astype(np.float32)
    y = np.random.randn(2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs


generated_inputs["jax.numpy.minimum"] = minimum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.minimum'.")


check_valid('jax.numpy.minimum', generated_inputs['jax.numpy.minimum'], lib="jax", suffix=0)
