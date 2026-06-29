
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic Python integers
    input_dict = {
        "x": 0,
        "loc": 0,
        "scale": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive offset
    input_dict = {
        "x": 5,
        "loc": 2,
        "scale": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values for x and loc
    input_dict = {
        "x": -2,
        "loc": -5,
        "scale": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int32 scalars
    input_dict = {
        "x": np.int32(10),
        "loc": np.int32(0),
        "scale": np.int32(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.int64 scalars
    input_dict = {
        "x": np.int64(-5),
        "loc": np.int64(5),
        "scale": np.int64(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger values
    input_dict = {
        "x": 100,
        "loc": 50,
        "scale": 20
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Standard logistic distribution parameters
    input_dict = {
        "x": 1,
        "loc": 0,
        "scale": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative x and positive loc
    input_dict = {
        "x": np.int32(-10),
        "loc": np.int32(10),
        "scale": np.int32(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale parameter
    input_dict = {
        "x": np.int64(0),
        "loc": np.int64(-100),
        "scale": np.int64(500)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal x and loc
    input_dict = {
        "x": 42,
        "loc": 42,
        "scale": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.cdf_3"] = jax_scipy_stats_logistic_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.cdf_3'.")


check_valid('jax.scipy.stats.logistic.cdf', generated_inputs['jax.scipy.stats.logistic.cdf_3'], lib="jax", suffix=3)
