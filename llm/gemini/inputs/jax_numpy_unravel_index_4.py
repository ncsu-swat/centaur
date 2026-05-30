
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Scalar index with 1D shape (using Python int and tuple of ints)
    indices = 5
    shape = (10,)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of indices with 2D shape (int32)
    indices = np.array([1, 3, 5], dtype=np.int32)
    shape = (2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of indices with 3D shape (int64)
    indices = np.array([[0, 10], [5, 20]], dtype=np.int64)
    shape = (3, 4, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative indices (supported in JAX)
    indices = np.array([-1, -3, -5], dtype=np.int32)
    shape = (4, 4)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Out of bounds indices (clipped in JAX)
    indices = np.array([100, -100], dtype=np.int32)
    shape = (3, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher-dimensional shape (4D)
    indices = np.array([10, 20, 30], dtype=np.int32)
    shape = (2, 3, 2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar numpy integer with 3D shape
    indices = np.int64(14)
    shape = (3, 3, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty indices array
    indices = np.array([], dtype=np.int32)
    shape = (5, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values
    indices = np.array([1000000, 2000000], dtype=np.int64)
    shape = (2000, 2000)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array of indices with 2D shape
    indices = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape = (4, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unravel_index_4"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_4'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_4'], lib="jax", suffix=4)
