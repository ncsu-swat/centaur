
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def abs_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32, negative
    x = np.int32(-42)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of int32, mixed signs
    x = np.array([-10, 0, 10, -20, 30], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of int64, mixed signs
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array of int16
    x = np.array([[[ -1,  2], [ -3,  4]], [[ -5,  6], [ -7,  8]]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Scalar int64, positive
    x = np.int64(100)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array of int8, including boundaries
    x = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array of int32
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D array of int16 containing all zeros
    x = np.zeros((3, 3), dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D array of int8
    x = np.random.randint(-10, 10, size=(2, 1, 3, 1, 2), dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array of int64, large values
    x = np.array([-922337203685477580, 922337203685477580], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.abs_2"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_2'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_2'], lib="jax", suffix=2)
