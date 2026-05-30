
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_less_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D integer array
    input_dict = {
        "x": 5,
        "y": np.array([1, 6, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with negative integers
    input_dict = {
        "x": 0,
        "y": np.array([[-1, 2], [0, -3]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float array with negative threshold
    input_dict = {
        "x": -2,
        "y": np.linspace(-5, 5, 10, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array
    input_dict = {
        "x": 10,
        "y": (np.random.randn(2, 3, 4) * 20).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int64 array with larger dimensions
    input_dict = {
        "x": -100,
        "y": np.arange(-105, -95).reshape(1, 2, 5, 1).astype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 zeros array
    input_dict = {
        "x": 1,
        "y": np.zeros((5, 5), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array
    input_dict = {
        "x": 10,
        "y": np.array([5, 15, 25], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array (scalar tensor)
    input_dict = {
        "x": 3,
        "y": np.array(4, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 array with random values
    input_dict = {
        "x": -5,
        "y": np.random.randint(-10, 10, size=(2, 2, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values in int64 array
    input_dict = {
        "x": 999999,
        "y": np.array([1000000, 500000, 2000000], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float32 array
    input_dict = {
        "x": 0,
        "y": np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.less_5"] = jax_numpy_less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_5'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_5'], lib="jax", suffix=5)
