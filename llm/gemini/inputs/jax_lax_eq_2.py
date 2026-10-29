
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def lax_eq_inputs():
    list_of_inputs = []

    # Input 1: Zero comparison
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unequal positive integers
    input_dict = {"x": 1, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Identical negative integers
    input_dict = {"x": -5, "y": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive and negative integers
    input_dict = {"x": 42, "y": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large equal positive integers
    input_dict = {"x": 100000, "y": 100000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero and negative integer
    input_dict = {"x": 0, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large unequal negative integers
    input_dict = {"x": -999999, "y": -123456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Distinct small integers
    input_dict = {"x": 7, "y": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large unequal positive integers
    input_dict = {"x": 12345, "y": 54321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and zero
    input_dict = {"x": -100, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Identical positive prime integers
    input_dict = {"x": 17, "y": 17}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.eq_2"] = lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_2'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_2'], lib="jax", suffix=2)
