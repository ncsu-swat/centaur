
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polar_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, float32, side='right'
    a = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square matrix, float32, side='left'
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-5,
        "max_iterations": 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (m > n), float64, side='right'
    a = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-7,
        "max_iterations": 20
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix (m < n), float32, side='left'
    a = np.random.randn(2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 30
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched matrices (3D), float32, side='right'
    a = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-8,
        "max_iterations": 150
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large batched matrices (4D), float64, side='left'
    a = np.random.randn(2, 2, 4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-9,
        "max_iterations": 200
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex matrix, side='right'
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 80
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with negative values, float32, side='left'
    a = -np.abs(np.random.randn(5, 5).astype(np.float32))
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-5,
        "max_iterations": 40
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-square matrix with float64, side='right'
    a = np.random.randn(6, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 120
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 matrix, float32, side='left'
    a = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-4,
        "max_iterations": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.polar"] = polar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.polar' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.polar'.")


check_valid('jax.scipy.linalg.polar', generated_inputs['jax.scipy.linalg.polar'], lib="jax", suffix=0)
