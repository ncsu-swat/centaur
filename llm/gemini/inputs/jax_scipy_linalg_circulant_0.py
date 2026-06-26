
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def circulant_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array (small length, standard)
    c = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 2: 1D float32 array with negative and zero values
    c = np.array([-1.5, 0.0, 2.5, -3.0], dtype=np.float32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 3: 1D float64 array of size 5
    c = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 4: 2D int32 array (batch of vectors)
    c = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 5: 2D float32 array (batch of size 3, vector size 4)
    c = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 6: 3D float32 array (multi-dimensional batch)
    c = np.random.randn(2, 2, 5).astype(np.float32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 7: 1D complex64 array
    c = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 8: 1D int16 array
    c = np.array([10, -20, 30], dtype=np.int16)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 9: 1D float32 array with single element (edge case N=1)
    c = np.array([42.0], dtype=np.float32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

    # Input 10: 1D float32 array with sequential values
    c = np.arange(10, dtype=np.float32)
    list_of_inputs.append({"c": copy.deepcopy(c)})

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
