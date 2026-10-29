
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def erfc_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive float
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative float
    input_dict = {"x": -0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive float
    input_dict = {"x": 3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative float
    input_dict = {"x": -3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive float
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small negative float
    input_dict = {"x": -1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Asymptotic positive float
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Asymptotic negative float
    input_dict = {"x": -10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Float representation of pi/2
    input_dict = {"x": 1.5707963267948966}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.erfc_3"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_3'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_3'], lib="jax", suffix=3)
