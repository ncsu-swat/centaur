
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 array, shape 5
    array = np.array(1.5, dtype=np.float32)
    shape = 5
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 2: 0D int32 array, shape 10
    array = np.array(-3, dtype=np.int32)
    shape = 10
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 3: 1D array of size 1, float64, shape 3
    array = np.array([2.71], dtype=np.float64)
    shape = 3
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 4: 1D array of size 1, int64, shape 8
    array = np.array([42], dtype=np.int64)
    shape = 8
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 5: 1D array of size 4, float32, shape 4
    array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shape = 4
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 6: 1D array of size 7, int8, shape 7
    array = np.array([-1, 0, 1, 2, -2, 3, -3], dtype=np.int8)
    shape = 7
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 7: 0D bool array, shape 2
    array = np.array(True, dtype=bool)
    shape = 2
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 8: 1D array of size 1, bool, shape 6
    array = np.array([False], dtype=bool)
    shape = 6
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 9: 1D array of size 2, float16, shape 2
    array = np.array([0.5, -0.5], dtype=np.float16)
    shape = 2
    list_of_inputs.append({"array": array, "shape": shape})

    # Input 10: 0D uint8 array, shape 15
    array = np.array(255, dtype=np.uint8)
    shape = 15
    list_of_inputs.append({"array": array, "shape": shape})

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
