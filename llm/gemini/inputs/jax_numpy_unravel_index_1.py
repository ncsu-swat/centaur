
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D indices, 2D shape
    indices = np.array([1, 3, 5], dtype=np.int32)
    shape = (2, 3)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 2: 1D indices, 3D shape
    indices = np.array([0, 7, 11], dtype=np.int32)
    shape = (2, 3, 2)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 3: Scalar index (0D array), 2D shape
    indices = np.array(5, dtype=np.int32)
    shape = (3, 3)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 4: Negative indices (supported by JAX), 2D shape
    indices = np.array([-1, -3, 2], dtype=np.int32)
    shape = (4, 4)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 5: 2D array of indices, 2D shape, int64
    indices = np.array([[1, 2], [3, 4]], dtype=np.int64)
    shape = (3, 5)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 6: 1D indices, 4D shape
    indices = np.array([10, 20, 30], dtype=np.int32)
    shape = (2, 2, 2, 5)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 7: Large indices with negative values, 3D shape
    indices = np.array([100, -50, 12], dtype=np.int64)
    shape = (10, 10, 10)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 8: 1D shape (trivial case)
    indices = np.array([0, 5, 9], dtype=np.int32)
    shape = (10,)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 9: Out-of-bounds indices (JAX clips these)
    indices = np.array([5, 15, 25], dtype=np.int32)
    shape = (5, 2)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    # Input 10: High-dimensional indices tensor, 2D shape
    indices = np.random.randint(-10, 20, size=(3, 3, 3)).astype(np.int32)
    shape = (4, 3)
    list_of_inputs.append({"indices": copy.deepcopy(indices), "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.unravel_index_1"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_1'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_1'], lib="jax", suffix=1)
