
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive float
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative float
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Moderate positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive float
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative float
    input_dict = {"x": -10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive float
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small negative float
    input_dict = {"x": -1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very large positive float
    input_dict = {"x": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very large negative float
    input_dict = {"x": -100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Arbitrary float value
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_2"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_2'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_2'], lib="jax", suffix=2)
