
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D vector norm (L2)
    x = np.random.randn(10).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (0,),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrix 1-norm with keepdims
    x = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (0, 1),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, vector norm along axis 1 (L0 norm)
    x = np.random.randn(4, 8).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 0,
        "axis": (1,),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, matrix L2 norm (largest singular value)
    x = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (1, 2),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, matrix negative 1-norm with keepdims
    x = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": -1,
        "axis": (0, 1),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, matrix negative 2-norm (smallest singular value)
    # Ensure matrix is non-singular by adding identity
    x = (np.random.randn(4, 4) + np.eye(4) * 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": -2,
        "axis": (0, 1),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, vector L1 norm along axis 2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (2,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, L1 norm with float64
    x = np.random.randn(15).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (0,),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, vector L2 norm along axis 1 with keepdims
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (1,),
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, matrix 1-norm along axes (0, 2)
    x = np.random.randn(3, 5, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (0, 2),
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_2"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_2'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_2'], lib="jax", suffix=2)
