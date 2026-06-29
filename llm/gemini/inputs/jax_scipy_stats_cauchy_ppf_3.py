
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_ppf_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar integers
    input_dict = {
        "q": int(0),
        "loc": int(0),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar numpy integers
    input_dict = {
        "q": np.int32(1),
        "loc": np.int32(5),
        "scale": np.int32(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array for q, scalar loc and scale
    input_dict = {
        "q": np.array([0, 1, 0], dtype=np.int32),
        "loc": int(-1),
        "scale": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array for q, scalar loc and scale
    input_dict = {
        "q": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "loc": int(10),
        "scale": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D arrays for all parameters
    input_dict = {
        "q": np.array([0, 1, 1], dtype=np.int32),
        "loc": np.array([-2, 0, 2], dtype=np.int32),
        "scale": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with different shapes (2D q, 1D loc, scalar scale)
    input_dict = {
        "q": np.array([[0, 0], [1, 1]], dtype=np.int32),
        "loc": np.array([1, 2], dtype=np.int32),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large integer values
    input_dict = {
        "q": np.int64(0),
        "loc": np.int64(1000),
        "scale": np.int64(500)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of zeros and ones for q
    input_dict = {
        "q": np.zeros((2, 2, 2), dtype=np.int32),
        "loc": int(-5),
        "scale": int(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-trivial broadcasting with 2D arrays
    input_dict = {
        "q": np.ones((3, 1), dtype=np.int64),
        "loc": np.zeros((1, 3), dtype=np.int64),
        "scale": np.ones((3, 3), dtype=np.int64) * 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixing int32 and int64
    input_dict = {
        "q": np.array([0, 1], dtype=np.int32),
        "loc": np.array([10, -10], dtype=np.int64),
        "scale": np.array([2, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.ppf_3"] = jax_scipy_stats_cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_3'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_3'], lib="jax", suffix=3)
