
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Cauchy evaluation at 0
    input_dict = {"x": 0, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, standard loc and scale
    input_dict = {"x": 1, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, scale > 1
    input_dict = {"x": -1, "loc": 0, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive x, non-zero loc
    input_dict = {"x": 5, "loc": 2, "scale": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative x and negative loc
    input_dict = {"x": -10, "loc": -5, "scale": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive x, loc, and scale
    input_dict = {"x": 100, "loc": 50, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero x, negative loc, scale > 1
    input_dict = {"x": 0, "loc": -10, "scale": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative x, positive loc, moderate scale
    input_dict = {"x": -20, "loc": 20, "scale": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x equals loc (mode of Cauchy distribution)
    input_dict = {"x": 42, "loc": 42, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extreme negative x with large scale
    input_dict = {"x": -999, "loc": 0, "scale": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive x with negative loc and large scale
    input_dict = {"x": 7, "loc": -3, "scale": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logpdf_3"] = cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_3'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_3'], lib="jax", suffix=3)
