
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def round_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"a": 1.532, "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"a": 3.267, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float, halfway rounding
    input_dict = {"a": -10.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive float, halfway rounding
    input_dict = {"a": 21.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero with non-zero decimals
    input_dict = {"a": 0.0, "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative small float with higher precision
    input_dict = {"a": -0.1234567, "decimals": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large float with decimals
    input_dict = {"a": 1234.5678, "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Halfway rounding to even (2.5 -> 2.0)
    input_dict = {"a": 2.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Halfway rounding to even (3.5 -> 4.0)
    input_dict = {"a": 3.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scientific notation value
    input_dict = {"a": 1e-5, "decimals": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative number with single decimal place
    input_dict = {"a": -999.999, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Pi rounded to 5 decimals
    input_dict = {"a": float(np.pi), "decimals": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.round_2"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.round_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.round_2'.")


check_valid('jax.numpy.round', generated_inputs['jax.numpy.round_2'], lib="jax", suffix=2)
