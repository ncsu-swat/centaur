
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def subtract_inputs():
    list_of_inputs = []

    # Input 1: float32 1D arrays of the same shape
    x = np.random.randn(10).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: int32 2D arrays of the same shape
    x = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    y = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: float64 3D arrays of the same shape
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting 2D and 1D arrays (float32)
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting 1D and 2D arrays (int64)
    x = np.random.randint(-50, 50, size=(5,)).astype(np.int64)
    y = np.random.randint(-50, 50, size=(2, 5)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Broadcasting scalar (0D array) and 2D array
    x = np.array(5.5, dtype=np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Broadcasting 3D arrays (int32)
    x = np.random.randint(-10, 10, size=(2, 1, 4)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(1, 3, 4)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: float32 4D arrays of the same shape
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: int32 1D arrays with negative values
    x = np.random.randint(-128, 127, size=(8,)).astype(np.int32)
    y = np.random.randint(-128, 127, size=(8,)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: float64 2D arrays with negative values
    x = np.random.randn(4, 4).astype(np.float64)
    y = np.random.randn(4, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Broadcasting 1D and scalar-like 0D array (float32)
    x = np.random.randn(10).astype(np.float32)
    y = np.array(-1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.subtract"] = subtract_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.subtract'.")


check_valid('jax.numpy.subtract', generated_inputs['jax.numpy.subtract'], lib="jax", suffix=0)
