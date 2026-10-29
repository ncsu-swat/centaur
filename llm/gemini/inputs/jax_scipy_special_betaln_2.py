
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def betaln_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integer floats
    input_dict = {
        "a": 1.0,
        "b": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Halves (symmetric)
    input_dict = {
        "a": 0.5,
        "b": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unequal positive integers as floats
    input_dict = {
        "a": 2.0,
        "b": 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small fractional values summing to 1
    input_dict = {
        "a": 0.1,
        "b": 0.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate fractional floats
    input_dict = {
        "a": 12.34,
        "b": 56.78
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Very small float and moderate float
    input_dict = {
        "a": 1e-5,
        "b": 2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large floats
    input_dict = {
        "a": 100.0,
        "b": 200.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative non-integer float for a, positive for b
    input_dict = {
        "a": -1.5,
        "b": 2.25
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Irrational-like constants (pi and e approximations)
    input_dict = {
        "a": 3.141592653589793,
        "b": 2.718281828459045
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Asymmetric scales (small and large)
    input_dict = {
        "a": 0.01,
        "b": 1000.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative non-integer float with sum positive
    input_dict = {
        "a": -0.25,
        "b": 1.75
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Positive fractional floats
    input_dict = {
        "a": 4.5,
        "b": 0.75
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.betaln_2"] = betaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.betaln_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.betaln_2'.")


check_valid('jax.scipy.special.betaln', generated_inputs['jax.scipy.special.betaln_2'], lib="jax", suffix=2)
