
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_invert_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    x = np.array([1, -2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D int32 array
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array
    x = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int64 array
    x = np.array([-100, 200, -300], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D int64 array
    x = np.array([[123456789, -987654321]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D bool array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D bool array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 0D int32 array
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 0D int64 array
    x = np.array(999999, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 0D bool array
    x = np.array(True, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_invert"] = bitwise_invert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_invert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_invert'.")


check_valid('jax.numpy.bitwise_invert', generated_inputs['jax.numpy.bitwise_invert'], lib="jax", suffix=0)
