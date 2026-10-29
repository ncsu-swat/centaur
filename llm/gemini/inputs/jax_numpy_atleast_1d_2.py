
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {"arys": np.array([1.0, 2.0, 3.5], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values
    input_dict = {"arys": np.array([-10, 0, 15, -20], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    input_dict = {
        "arys": np.array([[1.0, -2.5], [3.2, 4.8]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array
    input_dict = {
        "arys": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D empty float32 array
    input_dict = {"arys": np.array([], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D boolean array
    input_dict = {"arys": np.array([True, False, True], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array
    input_dict = {
        "arys": np.array([0.5, -1.5, 2.25, -3.75], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array
    input_dict = {
        "arys": np.zeros((2, 3, 4), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int16 array spanning boundary values
    input_dict = {
        "arys": np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with wide range
    input_dict = {
        "arys": np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D uint8 array
    input_dict = {"arys": np.array([0, 127, 255], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D single-element float64 array
    input_dict = {"arys": np.array([42.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_1d_2"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_2'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_2'], lib="jax", suffix=2)
