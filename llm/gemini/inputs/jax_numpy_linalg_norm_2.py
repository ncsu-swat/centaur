
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 1-norm along axis 0
    x = np.array([3.0, -4.0, 12.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (0,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, 2-norm along axis 0 with keepdims
    x = np.array([-1.5, 2.0, -3.5, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (0,),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, 0-norm (sparsity) along axis 1
    x = np.array([[0.0, 2.0, 0.0], [1.0, -3.0, 5.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "ord": 0,
        "axis": (1,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, 3-norm along axis 0 with keepdims
    x = np.random.randn(5, 4).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": 3,
        "axis": (0,),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, vector 1-norm along last axis
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (2,),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D matrix, 1-norm (max column sum)
    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (0, 1),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D matrix, -1-norm (min column sum) with keepdims
    x = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": -1,
        "axis": (0, 1),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D matrix, 2-norm (largest singular value)
    x = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (0, 1),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D square matrix, -2-norm (smallest singular value) with keepdims
    x = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    input_dict = {
        "x": x,
        "ord": -2,
        "axis": (0, 1),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D batched matrix, 2-norm along axes (1, 2)
    x = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "ord": 2,
        "axis": (1, 2),
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D complex array, matrix 1-norm along axes (-2, -1) with keepdims
    x = (
        np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)
    ).astype(np.complex64)
    input_dict = {
        "x": x,
        "ord": 1,
        "axis": (-2, -1),
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D batched array, matrix -1-norm along axes (-2, -1)
    x = np.random.randn(2, 2, 4, 4).astype(np.float64)
    input_dict = {
        "x": x,
        "ord": -1,
        "axis": (-2, -1),
        "keepdims": False,
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
