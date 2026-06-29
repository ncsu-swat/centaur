
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def norm_inputs():
    list_of_inputs = []

    # Input 1: 2D array, Frobenius norm, axis (0, 1), keepdims=False
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "fro",
        "axis": (0, 1),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, Nuclear norm, axis (0, 1), keepdims=True
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "nuc",
        "axis": (0, 1),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, Frobenius norm along axis (1, 2)
    x = np.random.randn(2, 3, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": "fro",
        "axis": (1, 2),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, Nuclear norm along axis (1, 2)
    x = np.random.randn(3, 5, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "nuc",
        "axis": (1, 2),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, Frobenius norm along axis (2, 3)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "fro",
        "axis": (2, 3),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, Nuclear norm along axis (0, 1)
    x = np.random.randn(4, 3, 2).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": "nuc",
        "axis": (0, 1),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative values, Frobenius norm, axis (-2, -1)
    x = np.random.uniform(-5, 5, (5, 5)).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "fro",
        "axis": (-2, -1),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, Nuclear norm, axis (0, 2)
    x = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": "nuc",
        "axis": (0, 2),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, Frobenius norm, axis (0, 3)
    x = np.random.randn(3, 3, 3, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": "fro",
        "axis": (0, 3),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, Nuclear norm, axis (0, 1), float64
    x = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": "nuc",
        "axis": (0, 1),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_4"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_4'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_4'], lib="jax", suffix=4)
