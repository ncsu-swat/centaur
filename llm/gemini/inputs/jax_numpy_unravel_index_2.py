
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_unravel_index_2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D indices, standard shape
    indices = np.array([1, 2, 3], dtype=np.int32)
    shape = 5
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 2: 2D indices, larger shape
    indices = np.array([[0, 2], [4, 6]], dtype=np.int32)
    shape = 10
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 3: Negative indices (supported in JAX)
    indices = np.array([-1, -3, -5], dtype=np.int32)
    shape = 8
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 4: Out-of-bound indices
    indices = np.array([10, -10, 20], dtype=np.int32)
    shape = 6
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 5: Large int64 indices
    indices = np.array([1000, 5000, 9999], dtype=np.int64)
    shape = 10000
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 6: 0D tensor (scalar index)
    indices = np.array(4, dtype=np.int32)
    shape = 12
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 7: int32 indices with different values
    indices = np.array([2, 5, 8], dtype=np.int32)
    shape = 15
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 8: 3D indices
    indices = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape = 20
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 9: Edge case - shape = 1
    indices = np.array([0, 0, 0], dtype=np.int32)
    shape = 1
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 10: Empty indices tensor
    indices = np.array([], dtype=np.int32)
    shape = 3
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.unravel_index_2"] = jax_numpy_unravel_index_2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_2'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_2'], lib="jax", suffix=2)
