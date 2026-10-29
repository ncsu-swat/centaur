
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 2-norm (vector norm), 1-tuple axis
    x = np.array([3.0, -4.0, 12.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "ord": 2.0,
        "axis": (0,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, 1-norm along axis 0, keepdims=True
    x = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1.0,
        "axis": (0,),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, infinity norm along axis 1
    x = np.random.randn(3, 6).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": float("inf"),
        "axis": (1,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, matrix 1-norm over (0, 1)
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1.0,
        "axis": (0, 1),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, matrix infinity norm with keepdims=True
    x = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": float("inf"),
        "axis": (0, 1),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, matrix spectral norm (ord=2.0)
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": 2.0,
        "axis": (0, 1),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, vector norm ord=3.0 along axis 2
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 3.0,
        "axis": (2,),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, matrix norm over axes (1, 2)
    x = np.random.randn(2, 4, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1.0,
        "axis": (1, 2),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, matrix norm with ord=-1.0 over axes (0, 2)
    x = np.random.randn(3, 2, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": -1.0,
        "axis": (0, 2),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, vector norm with negative infinity ord
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": float("-inf"),
        "axis": (3,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, matrix norm with ord=2.0 over axes (2, 3)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2.0,
        "axis": (2, 3),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D array, vector 0-norm (sparsity) along axis 0
    x = np.array([[0.0, 1.0, -2.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "ord": 0.0,
        "axis": (0,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.norm_6"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_6'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_6'], lib="jax", suffix=6)
