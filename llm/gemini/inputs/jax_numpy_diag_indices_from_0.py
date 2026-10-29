
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def diag_indices_from_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix of float32
    arr = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 2: 2D square matrix of int32 with negative values
    arr = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 3: 3D array with equal dimensions (2, 2, 2)
    arr = np.ones((2, 2, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 4: 4D array with equal dimensions (3, 3, 3, 3)
    arr = np.arange(81, dtype=np.int64).reshape((3, 3, 3, 3))
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 5: 2D matrix of shape (1, 1)
    arr = np.array([[42.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 6: 3D array of shape (1, 1, 1) with bool dtype
    arr = np.array([[[True]]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 7: 2D complex array of shape (4, 4)
    arr = np.eye(4, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 8: 5D array of shape (2, 2, 2, 2, 2)
    arr = np.zeros((2, 2, 2, 2, 2), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 9: 2D array of shape (10, 10) float16
    arr = np.random.randn(10, 10).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 10: 3D array of shape (4, 4, 4) with int16
    arr = np.arange(64, dtype=np.int16).reshape((4, 4, 4))
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    # Input 11: 2D array of shape (0, 0)
    arr = np.empty((0, 0), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr}))

    return list_of_inputs


generated_inputs["jax.numpy.diag_indices_from"] = diag_indices_from_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diag_indices_from' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diag_indices_from'.")


check_valid('jax.numpy.diag_indices_from', generated_inputs['jax.numpy.diag_indices_from'], lib="jax", suffix=0)
