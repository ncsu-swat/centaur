
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diag_indices_from_inputs():
    list_of_inputs = []

    # Input 1: 2D array of shape (2, 2), int32
    arr = np.arange(4, dtype=np.int32).reshape(2, 2)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 2: 2D array of shape (3, 3), float32
    arr = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 3: 2D array of shape (10, 10), int64
    arr = np.zeros((10, 10), dtype=np.int64)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 4: 3D array of shape (2, 2, 2), float64
    arr = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 5: 3D array of shape (4, 4, 4), int32
    arr = np.ones((4, 4, 4), dtype=np.int32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 6: 4D array of shape (3, 3, 3, 3), float32
    arr = np.random.randn(3, 3, 3, 3).astype(np.float32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 7: 2D array of shape (1, 1), int32
    arr = np.array([[5]], dtype=np.int32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 8: 5D array of shape (2, 2, 2, 2, 2), float32
    arr = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 9: 3D array of shape (5, 5, 5), int64
    arr = np.ones((5, 5, 5), dtype=np.int64)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

    # Input 10: 2D array of shape (100, 100), float32
    arr = np.random.randn(100, 100).astype(np.float32)
    list_of_inputs.append({"arr": copy.deepcopy(arr)})

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
