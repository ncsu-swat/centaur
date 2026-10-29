
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def zeta_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"x": 2.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"x": 3.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"x": 2.0, "q": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"x": 4.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"x": 3.0, "q": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"x": 5.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"x": 2.0, "q": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"x": 4.0, "q": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"x": 6.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"x": 3.0, "q": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {"x": 5.0, "q": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {"x": 2.0, "q": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.zeta_3"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_3'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_3'], lib="jax", suffix=3)
