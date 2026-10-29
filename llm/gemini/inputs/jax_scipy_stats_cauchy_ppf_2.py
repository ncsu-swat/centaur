
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def cauchy_ppf_inputs():
    list_of_inputs = []

    # Input 1: Standard median
    input_dict = {
        "q": 0.5,
        "loc": 0.0,
        "scale": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower quartile
    input_dict = {
        "q": 0.25,
        "loc": 0.0,
        "scale": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Upper quartile
    input_dict = {
        "q": 0.75,
        "loc": 0.0,
        "scale": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative location, scale > 1
    input_dict = {
        "q": 0.1,
        "loc": -5.0,
        "scale": 2.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive location, small scale
    input_dict = {
        "q": 0.9,
        "loc": 10.0,
        "scale": 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boundary near zero
    input_dict = {
        "q": 0.001,
        "loc": 0.0,
        "scale": 0.1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boundary near one
    input_dict = {
        "q": 0.999,
        "loc": -2.0,
        "scale": 3.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small scale
    input_dict = {
        "q": 0.4,
        "loc": 1.23,
        "scale": 1e-4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large location and scale
    input_dict = {
        "q": 0.6,
        "loc": 1000.0,
        "scale": 500.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extreme tails (0.0)
    input_dict = {
        "q": 0.0,
        "loc": 0.0,
        "scale": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Extreme tails (1.0)
    input_dict = {
        "q": 1.0,
        "loc": 0.0,
        "scale": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.ppf_2"] = cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_2'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_2'], lib="jax", suffix=2)
