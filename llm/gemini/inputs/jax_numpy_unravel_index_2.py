
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array with int32
    indices = np.array(3, dtype=np.int32)
    shape = 5
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 2: 1D array with standard positive indices
    indices = np.array([0, 2, 4], dtype=np.int32)
    shape = 6
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 3: 1D array with negative indices
    indices = np.array([-1, -3, -5], dtype=np.int32)
    shape = 5
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 4: 1D array with out-of-bound values (int64)
    indices = np.array([10, 50, -20], dtype=np.int64)
    shape = 10
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 5: 2D array of indices
    indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    shape = 10
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 6: 2D array of zeros with minimal shape
    indices = np.zeros((3, 3), dtype=np.int64)
    shape = 1
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 7: 3D array of indices
    indices = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape = 20
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 8: Empty 1D array
    indices = np.array([], dtype=np.int32)
    shape = 8
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 9: 1D array with int32 dtype and consecutive values
    indices = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = 7
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 10: 4D array with int64 dtype
    indices = np.ones((1, 2, 1, 3), dtype=np.int64)
    shape = 15
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 11: 1D array with int32 dtype including negative and zero
    indices = np.array([-5, 0, 5], dtype=np.int32)
    shape = 12
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 12: 1D array with int64 dtype and large shape
    indices = np.array([0, 10, 20, 30], dtype=np.int64)
    shape = 50
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    return list_of_inputs


generated_inputs["jax.numpy.unravel_index_2"] = unravel_index_inputs()

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
