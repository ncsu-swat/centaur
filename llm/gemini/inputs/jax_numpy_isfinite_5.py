
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with finite, inf, and nan
    input_dict = {"x": np.array([1.0, 2.0, np.inf, np.nan], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array with infinities and zero
    input_dict = {"x": np.array([-np.inf, 0.0, 1e5, np.inf], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D integer array
    input_dict = {"x": np.array([-10, 0, 5, 42], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array
    input_dict = {"x": np.array([True, False, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array with nan
    input_dict = {"x": np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array with positive and negative infinities
    input_dict = {
        "x": np.array([[np.inf, 1.0, -1.0], [-np.inf, 0.0, np.nan]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array
    input_dict = {
        "x": np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, np.nan], [7.0, np.inf]]],
            dtype=np.float32,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex64 array
    input_dict = {
        "x": np.array(
            [1.0 + 2.0j, np.nan + 1.0j, 3.0 + np.inf * 1j, 4.0 - 5.0j],
            dtype=np.complex64,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D single-element array
    input_dict = {"x": np.array([-np.inf], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array with negative numbers
    input_dict = {"x": np.array([-100, -1, 0, 1, 100], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D column vector
    input_dict = {"x": np.array([[1.5], [-2.5], [np.nan]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isfinite_5"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_5'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_5'], lib="jax", suffix=5)
