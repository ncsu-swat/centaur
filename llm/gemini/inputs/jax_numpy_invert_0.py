
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def invert_inputs():
    list_of_inputs = []

    # Input 1: boolean 1D array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: int32 1D array
    x = np.array([0, 1, 2, 128, 255], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: int32 2D array with negative and positive values
    x = np.array([[-128, -1, 0], [1, 127, -50]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: int32 3D array
    x = np.array([[[100, -200], [300, -400]], [[500, -600], [700, -800]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: int64 1D array
    x = np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: int32 4D array
    x = np.zeros((2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: int64 2D array
    x = np.ones((3, 3), dtype=np.int64) * 5
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: int64 3D array
    x = np.array([[[1000, -1000]]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: boolean 2D array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: int32 0D array (scalar)
    x = np.array(-5, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: boolean 3D array
    x = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.invert"] = invert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.invert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.invert'.")


check_valid('jax.numpy.invert', generated_inputs['jax.numpy.invert'], lib="jax", suffix=0)
