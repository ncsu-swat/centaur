
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def circulant_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    c = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 2: 1D float32 array with negative values
    c = np.array([-1.5, 0.0, 2.5, -3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 3: 1D float64 array
    c = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 4: 1D complex64 array
    c = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 5: 1D complex128 array
    c = np.array([2.5 - 1.5j, -0.5 + 3.2j, 0.0 + 0.0j, 1.1 + 1.1j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 6: 1D single element array
    c = np.array([42], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 7: 2D array (batch of 1D vectors)
    c = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 8: 2D float32 array with negative values
    c = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 9: 3D array (higher-order batching)
    c = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 10: 4D array with boolean/uint8 values
    c = np.random.randint(0, 10, size=(2, 2, 3, 6)).astype(np.uint8)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    # Input 11: 1D boolean array
    c = np.array([True, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"c": c}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.circulant"] = circulant_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.circulant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.circulant'.")


check_valid('jax.scipy.linalg.circulant', generated_inputs['jax.scipy.linalg.circulant'], lib="jax", suffix=0)
