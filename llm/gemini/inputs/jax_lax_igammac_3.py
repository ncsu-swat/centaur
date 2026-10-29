
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def igammac_inputs():
    list_of_inputs = []

    # Input 1: Basic values
    input_dict = {"a": 1.0, "x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: a > x
    input_dict = {"a": 5.0, "x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: a < x
    input_dict = {"a": 2.0, "x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boundary condition x = 0.0
    input_dict = {"a": 3.0, "x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Equal values
    input_dict = {"a": 10.0, "x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-integer float values
    input_dict = {"a": 0.5, "x": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive parameters
    input_dict = {"a": 0.1, "x": 0.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger values
    input_dict = {"a": 50.0, "x": 30.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x substantially larger than a
    input_dict = {"a": 4.0, "x": 25.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: a substantially larger than x
    input_dict = {"a": 30.0, "x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Fractional values
    input_dict = {"a": 2.5, "x": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Moderate values
    input_dict = {"a": 8.0, "x": 12.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.igammac_3"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igammac_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igammac_3'.")


check_valid('jax.lax.igammac', generated_inputs['jax.lax.igammac_3'], lib="jax", suffix=3)
