
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pinv_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, rtol specified
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-5,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular float32 matrix (M > N), rcond specified
    a = np.random.randn(6, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": None,
        "rcond": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D symmetric float64 matrix, hermitian=True, rtol specified
    a_raw = np.random.randn(4, 4).astype(np.float64)
    a = a_raw + a_raw.T
    input_dict = {
        "a": a,
        "rtol": 1e-12,
        "hermitian": True,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched float32 matrices, rtol specified
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-6,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D batched float32 matrices, rcond specified
    a = np.random.randn(2, 2, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": None,
        "rcond": 1e-4,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D symmetric float32 matrix, hermitian=True, rcond specified
    a_raw = np.random.randn(8, 8).astype(np.float32)
    a = a_raw + a_raw.T
    input_dict = {
        "a": a,
        "rtol": None,
        "rcond": 1e-7,
        "hermitian": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D rectangular float64 matrix (M < N), rtol specified
    a = np.random.randn(3, 7).astype(np.float64)
    input_dict = {
        "a": a,
        "rtol": 1e-10,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D batched symmetric float64 matrices, hermitian=True
    a_raw = np.random.randn(3, 4, 4).astype(np.float64)
    a = a_raw + np.swapaxes(a_raw, -1, -2)
    input_dict = {
        "a": a,
        "rtol": 1e-8,
        "hermitian": True,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 matrix with negative values, hermitian=False
    a = np.random.uniform(-10, -1, size=(4, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-5,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D large float64 matrix, rcond specified
    a = np.random.randn(50, 50).astype(np.float64)
    input_dict = {
        "a": a,
        "rtol": None,
        "rcond": 1e-12,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.pinv_1"] = pinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.pinv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.pinv_1'.")


check_valid('jax.numpy.linalg.pinv', generated_inputs['jax.numpy.linalg.pinv_1'], lib="jax", suffix=1)
