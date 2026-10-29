
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fiedler_inputs():
    list_of_inputs = []

    # Input 1: 1D positive integer array
    a = np.array([1, 4, 12, 45, 77], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 1D float32 array with negative and positive values
    a = np.array([-2.5, 0.0, 1.5, 3.2, -4.1], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 1D float64 array
    a = np.array([0.1, 0.5, 1.2, 2.3], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 1D array of length 1 (scalar vector)
    a = np.array([5.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 1D empty array
    a = np.array([], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 2D array (batched 1D arrays)
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 2D array with float values and negative numbers
    a = np.array([[-1.0, 2.0, -3.0], [0.5, -0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 3D array (batch of vectors)
    a = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 4D array
    a = np.random.randn(2, 2, 2, 5).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 1D large integer array
    a = np.arange(20, dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 11: 1D array with duplicate values
    a = np.array([2.0, 2.0, 5.0, 5.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs


generated_inputs["jax.scipy.linalg.fiedler"] = fiedler_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.fiedler' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.fiedler'.")


check_valid('jax.scipy.linalg.fiedler', generated_inputs['jax.scipy.linalg.fiedler'], lib="jax", suffix=0)
