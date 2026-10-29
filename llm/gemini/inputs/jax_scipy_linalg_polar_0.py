
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polar_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D square matrix with method="svd"
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix with side="left" and method="svd"
    a = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-7,
        "max_iterations": 15,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix m > n, right decomposition with method="svd"
    a = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-5,
        "max_iterations": 20,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix m < n, left decomposition with method="svd"
    a = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 25,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched matrices (3D tensor), right decomposition with method="svd"
    a = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension batch (4D tensor), method="svd"
    a = np.random.randn(2, 2, 4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-8,
        "max_iterations": 30,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex valued matrix, method="svd"
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-5,
        "max_iterations": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex valued rectangular matrix, method="svd"
    a = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "side": "left",
        "method": "svd",
        "eps": 1e-9,
        "max_iterations": 15,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values, rectangular m > n with method="svd"
    a = -np.abs(np.random.randn(6, 2).astype(np.float64))
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 12,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched rectangular matrices m < n, right decomposition with method="svd"
    a = np.random.randn(3, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "side": "right",
        "method": "svd",
        "eps": 1e-6,
        "max_iterations": 15,
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
