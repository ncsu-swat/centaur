
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def expm1_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer-valued float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer-valued float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive value
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative value
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Very small positive value near machine precision
    input_dict = {"x": 1e-10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate positive float
    input_dict = {"x": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Moderate negative float
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive float
    input_dict = {"x": 15.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative float
    input_dict = {"x": -20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Fractional float
    input_dict = {"x": 0.123456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.expm1_2"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_2'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_2'], lib="jax", suffix=2)
