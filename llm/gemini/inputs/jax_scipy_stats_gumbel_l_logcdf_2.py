
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Standard parameters
    input_dict = {
        "x": 1.0,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative x value
    input_dict = {
        "x": -1.5,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive shifted distribution
    input_dict = {
        "x": 5.0,
        "loc": 2.0,
        "scale": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large scale parameter
    input_dict = {
        "x": 0.0,
        "loc": -1.0,
        "scale": 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small scale parameter
    input_dict = {
        "x": -2.0,
        "loc": -3.0,
        "scale": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large x relative to loc
    input_dict = {
        "x": 15.0,
        "loc": 2.0,
        "scale": 3.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small x (negative infinity direction)
    input_dict = {
        "x": -20.0,
        "loc": 0.0,
        "scale": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float inputs using np.float64
    input_dict = {
        "x": float(np.float64(2.5)),
        "loc": float(np.float64(1.0)),
        "scale": float(np.float64(2.0))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float inputs using np.float32 cast to Python float
    input_dict = {
        "x": float(np.float32(-0.5)),
        "loc": float(np.float32(0.5)),
        "scale": float(np.float32(0.1))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal x and loc
    input_dict = {
        "x": 3.14,
        "loc": 3.14,
        "scale": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logcdf_2"] = gumbel_l_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logcdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logcdf_2'.")


check_valid('jax.scipy.stats.gumbel_l.logcdf', generated_inputs['jax.scipy.stats.gumbel_l.logcdf_2'], lib="jax", suffix=2)
