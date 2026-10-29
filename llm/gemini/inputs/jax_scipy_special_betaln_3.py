
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.scipy.special

def betaln_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {"a": 1, "b": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small distinct integers
    input_dict = {"a": 2, "b": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger first argument
    input_dict = {"a": 5, "b": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Equal moderate integers
    input_dict = {"a": 10, "b": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger integers
    input_dict = {"a": 100, "b": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Prime integers
    input_dict = {"a": 7, "b": 13}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: a=1 with larger b
    input_dict = {"a": 1, "b": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger a with b=1
    input_dict = {"a": 20, "b": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Moderate asymmetric integers
    input_dict = {"a": 15, "b": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High magnitude integers
    input_dict = {"a": 250, "b": 500}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Consecutive integers
    input_dict = {"a": 8, "b": 9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.betaln_3"] = betaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.betaln_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.betaln_3'.")


check_valid('jax.scipy.special.betaln', generated_inputs['jax.scipy.special.betaln_3'], lib="jax", suffix=3)
