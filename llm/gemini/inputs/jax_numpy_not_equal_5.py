
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: x is 0, y is a 1D integer array
    input_dict = {
        "x": 0,
        "y": np.array([0, 1, -1, 0, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x is 5, y is a 2D float32 array
    input_dict = {
        "x": 5,
        "y": np.array([[5.0, 1.2, 5.0], [3.4, 5.0, 6.7]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x is -1, y is a 3D int64 array
    input_dict = {
        "x": -1,
        "y": np.arange(-5, 7).reshape(2, 3, 2).astype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: x is 1, y is a 1D boolean array
    input_dict = {
        "x": 1,
        "y": np.array([True, False, True], dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x is 42, y is a 0D scalar-like array
    input_dict = {
        "x": 42,
        "y": np.array(42, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x is -100, y is a 2D uint8 array
    input_dict = {
        "x": -100,
        "y": np.array([[0, 100], [156, 255]], dtype=np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x is 0, y is a 1D float64 array with zeros
    input_dict = {
        "x": 0,
        "y": np.zeros(10, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x is 10, y is a 4D array of random integers
    input_dict = {
        "x": 10,
        "y": np.random.randint(0, 20, size=(2, 2, 2, 2)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x is -3, y is a 1D float16 array
    input_dict = {
        "x": -3,
        "y": np.array([-3.0, 0.0, 3.0], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x is 2, y is a 5D array of twos
    input_dict = {
        "x": 2,
        "y": np.ones((1, 2, 1, 3, 1), dtype=np.int32) * 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_5"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_5'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_5'], lib="jax", suffix=5)
