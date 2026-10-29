
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pinv_inputs():
    list_of_inputs = []

    # Input 1: 2D rectangular matrix (M > N), float32, hermitian=False with rtol
    a = np.random.randn(4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-4,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular matrix (M < N), float64, hermitian=False with rtol
    a = np.random.randn(3, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "rtol": 1e-5,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D square symmetric matrix, float32, hermitian=True with rtol
    a = np.random.randn(4, 4).astype(np.float32)
    a = (a + a.T) / 2.0
    input_dict = {
        "a": a,
        "rtol": 1e-4,
        "hermitian": True,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched rectangular matrices, float32, hermitian=False with rtol
    a = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-3,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched square symmetric matrices, float64, hermitian=True with rtol
    a = np.random.randn(3, 4, 4).astype(np.float64)
    a = (a + np.swapaxes(a, -1, -2)) / 2.0
    input_dict = {
        "a": a,
        "rtol": 1e-5,
        "hermitian": True,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D batched rectangular matrices, float32, hermitian=False with rtol
    a = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": 1e-4,
        "hermitian": False,
        "rcond": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D rectangular matrix (M > N), float32, hermitian=False with rcond
    a = np.random.randn(6, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "rtol": None,
        "hermitian": False,
        "rcond": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D square symmetric matrix, float64, hermitian=True with rcond
    a = np.random.randn(5, 5).astype(np.float64)
    a = (a + a.T) / 2.0
    input_dict = {
        "a": a,
        "rtol": None,
        "hermitian": True,
        "rcond": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D batched rectangular matrices, float64, hermitian=False with rcond
    a = np.random.randn(4, 2, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "rtol": None,
        "hermitian": False,
        "rcond": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D batched square symmetric matrices, float32, hermitian=True with rcond
    a = np.random.randn(2, 3, 3).astype(np.float32)
    a = (a + np.swapaxes(a, -1, -2)) / 2.0
    input_dict = {
        "a": a,
        "rtol": None,
        "hermitian": True,
        "rcond": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D batched square symmetric matrices, float32, hermitian=True with rcond
    a = np.random.randn(2, 1, 3, 3).astype(np.float32)
    a = (a + np.swapaxes(a, -1, -2)) / 2.0
    input_dict = {
        "a": a,
        "rtol": None,
        "hermitian": True,
        "rcond": 1e-5
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
