
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Standard values
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive shift and larger scale
    input_dict = {"x": 1.5, "loc": 0.5, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x and negative loc, small scale
    input_dict = {"x": -2.5, "loc": -1.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive x
    input_dict = {"x": 10.0, "loc": 5.0, "scale": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative x
    input_dict = {"x": -10.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Arbitrary fractional values
    input_dict = {"x": 0.123, "loc": 0.456, "scale": 0.789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very large scale and x
    input_dict = {"x": 100.0, "loc": 10.0, "scale": 15.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative x with positive loc and moderate scale
    input_dict = {"x": -0.5, "loc": 2.0, "scale": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Transcendental / mathematical constants
    input_dict = {"x": 3.14159, "loc": 2.71828, "scale": 1.41421}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Deep negative region
    input_dict = {"x": -50.0, "loc": -20.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small scale value
    input_dict = {"x": 0.0, "loc": 1.0, "scale": 0.1}
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
