
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is zero, x2 is a 1D float32 array
    input_dict = {"x1": 0.0, "x2": np.array([0.5, 1.0, 2.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x1 is negative, x2 is a 1D float64 array
    input_dict = {
        "x1": -3.5,
        "x2": np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x1 is positive, x2 is a 2D float32 array
    input_dict = {
        "x1": 1.25,
        "x2": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: x1 is zero, x2 is a 2D array with negative and positive values
    input_dict = {
        "x1": 0.0,
        "x2": np.array([[-0.5, 0.5], [1.5, -1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x1 is a large negative float, x2 is a 3D float32 array
    input_dict = {
        "x1": -100.0,
        "x2": np.ones((2, 3, 4), dtype=np.float32) * 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x1 is a large positive float, x2 is a 3D float64 array
    input_dict = {
        "x1": 50.0,
        "x2": np.zeros((3, 2, 2), dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 is zero, x2 is a 0D scalar array
    input_dict = {"x1": 0.0, "x2": np.array(0.75, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x1 is positive, x2 is a 4D array
    input_dict = {
        "x1": 0.01,
        "x2": np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(
            np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x1 is negative, x2 is an integer array
    input_dict = {
        "x1": -0.001,
        "x2": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x1 is zero, x2 is a 1D int64 array
    input_dict = {
        "x1": 0.0,
        "x2": np.array([0, 1, -1, 10], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x1 is positive, x2 is a 5D float32 array
    input_dict = {
        "x1": 2.5,
        "x2": np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_4"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_4'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_4'], lib="jax", suffix=4)
