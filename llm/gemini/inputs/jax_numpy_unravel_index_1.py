
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: 1D indices, 2D shape
    indices = np.array([1, 3, 5], dtype=np.int32)
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 2: 0D scalar array index, 2D shape
    indices = np.array(4, dtype=np.int32)
    shape = (3, 3)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 3: 1D indices with negative values, 3D shape
    indices = np.array([-1, 0, 5, -3], dtype=np.int64)
    shape = (2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 4: 2D array of indices, 2D shape
    indices = np.array([[0, 2], [3, 5]], dtype=np.int32)
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 5: 1D indices, 1D shape
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    shape = (5,)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 6: 1D indices with out-of-bound values, 2D shape
    indices = np.array([-10, 0, 5, 20], dtype=np.int32)
    shape = (3, 4)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 7: 3D indices array, 4D shape
    indices = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    shape = (2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 8: Empty indices array, 2D shape
    indices = np.array([], dtype=np.int32)
    shape = (4, 4)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 9: 1D int64 indices, 2D shape
    indices = np.array([0, 2, 4, 6], dtype=np.int64)
    shape = (3, 3)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 10: 2D indices with negative values, 3D shape
    indices = np.array([[-5, -1], [2, 10]], dtype=np.int64)
    shape = (2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

    # Input 11: 1D indices, 5D shape
    indices = np.array([0, 15, 31], dtype=np.int32)
    shape = (2, 2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"indices": indices, "shape": shape}))

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
