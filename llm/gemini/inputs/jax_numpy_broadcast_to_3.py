
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 array broadcast to size 5
    array = np.array(3.14, dtype=np.float32)
    shape = 5
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 2: 1D float32 array of shape (1,) broadcast to size 4
    array = np.array([-1.5], dtype=np.float32)
    shape = 4
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 3: 1D int32 array of shape (3,) broadcast to size 3
    array = np.array([10, -20, 30], dtype=np.int32)
    shape = 3
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 4: 0D int64 array broadcast to size 1
    array = np.array(-42, dtype=np.int64)
    shape = 1
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 5: 1D int64 array of shape (1,) broadcast to size 10
    array = np.array([100], dtype=np.int64)
    shape = 10
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 6: 0D float64 array broadcast to size 2
    array = np.array(-0.001, dtype=np.float64)
    shape = 2
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 7: 1D float64 array of shape (7,) broadcast to size 7
    array = np.arange(7, dtype=np.float64)
    shape = 7
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 8: 0D int32 array broadcast to size 8
    array = np.array(255, dtype=np.int32)
    shape = 8
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 9: 1D float32 array of shape (1,) broadcast to size 16
    array = np.array([0.0], dtype=np.float32)
    shape = 16
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 10: 1D int32 array of shape (1,) broadcast to size 6
    array = np.array([-5], dtype=np.int32)
    shape = 6
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_to_3"] = broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_to_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_to_3'.")


check_valid('jax.numpy.broadcast_to', generated_inputs['jax.numpy.broadcast_to_3'], lib="jax", suffix=3)
