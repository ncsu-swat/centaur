
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard positive float
    input_dict = {
        "x": 0.5,
        "y": np.array([0.1, 0.5, 0.9], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, negative float
    input_dict = {
        "x": -2.5,
        "y": np.array([-5, -3, -2, 0, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, zero float
    input_dict = {
        "x": 0.0,
        "y": np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array
    input_dict = {
        "x": 10.0,
        "y": np.array(5.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with negative and positive values
    input_dict = {
        "x": 1.23,
        "y": np.random.randn(2, 3, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array with float16 dtype
    input_dict = {
        "x": -0.01,
        "y": np.ones((2, 2, 2, 2), dtype=np.float16) * -0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64 array with large values
    input_dict = {
        "x": 1e9,
        "y": np.array([10**8, 10**9, 10**10], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with inf and nan values
    input_dict = {
        "x": 5.0,
        "y": np.array([[np.inf, -np.inf], [np.nan, 5.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D uint8 array
    input_dict = {
        "x": 128.0,
        "y": np.array([0, 50, 128, 200, 255], dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D boolean array
    input_dict = {
        "x": 0.5,
        "y": np.array([[True, False], [False, True]], dtype=bool),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D array
    input_dict = {
        "x": -100.0,
        "y": np.zeros((1, 2, 1, 3, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.less_4"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_4'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_4'], lib="jax", suffix=4)
