
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {
        "x": int(2),
        "loc": int(1),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero value for x, default-like loc and positive scale
    input_dict = {
        "x": int(0),
        "loc": int(0),
        "scale": int(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x and loc, positive scale
    input_dict = {
        "x": int(-5),
        "loc": int(-2),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive scale
    input_dict = {
        "x": int(10),
        "loc": int(5),
        "scale": int(100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.int32 types
    input_dict = {
        "x": np.int32(3),
        "loc": np.int32(-1),
        "scale": np.int32(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int64 types
    input_dict = {
        "x": np.int64(-10),
        "loc": np.int64(0),
        "scale": np.int64(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large x relative to loc
    input_dict = {
        "x": int(1000),
        "loc": int(10),
        "scale": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small x relative to loc
    input_dict = {
        "x": int(-1000),
        "loc": int(-10),
        "scale": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large loc
    input_dict = {
        "x": int(0),
        "loc": int(500),
        "scale": int(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All ones (positive scale)
    input_dict = {
        "x": int(1),
        "loc": int(1),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logcdf_3"] = gumbel_l_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logcdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logcdf_3'.")


check_valid('jax.scipy.stats.gumbel_l.logcdf', generated_inputs['jax.scipy.stats.gumbel_l.logcdf_3'], lib="jax", suffix=3)
