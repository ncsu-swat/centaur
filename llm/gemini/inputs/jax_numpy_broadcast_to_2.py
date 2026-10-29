
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar to 1-D
    array = np.array(42, dtype=np.int32)
    shape = [5]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 2: 1-D array to 2-D
    array = np.array([1, 2, 3], dtype=np.int64)
    shape = [4, 3]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 3: 2-D column vector to 2-D matrix
    array = np.array([[-1.5], [2.5], [-3.5]], dtype=np.float32)
    shape = [3, 4]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 4: 2-D row vector to 2-D matrix with negative values
    array = np.array([[-10, 20, -30, 40]], dtype=np.float64)
    shape = [5, 4]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 5: 1-D array broadcast to 3-D
    array = np.array([True, False], dtype=np.bool_)
    shape = [2, 3, 2]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 6: 2-D array broadcast to 3-D
    array = np.arange(6, dtype=np.int16).reshape(2, 3)
    shape = [4, 2, 3]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 7: 3-D array with singleton dimensions broadcast to 3-D
    array = np.array([[[1.0], [-2.0]]], dtype=np.float32)  # shape (1, 2, 1)
    shape = [3, 2, 5]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 8: 0-D complex scalar broadcast to 2-D
    array = np.array(1.0 - 2.0j, dtype=np.complex64)
    shape = [2, 2]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 9: 1-D array broadcast to 4-D
    array = np.array([0, 1], dtype=np.uint8)
    shape = [2, 3, 4, 2]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 10: Array broadcast to the same shape (identity broadcast)
    array = np.random.randn(3, 4).astype(np.float32)
    shape = [3, 4]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 11: 4-D array with singleton dimensions broadcast to 4-D
    array = np.ones((1, 4, 1, 3), dtype=np.int32)
    shape = [2, 4, 5, 3]
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_to_2"] = broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_to_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_to_2'.")


check_valid('jax.numpy.broadcast_to', generated_inputs['jax.numpy.broadcast_to_2'], lib="jax", suffix=2)
