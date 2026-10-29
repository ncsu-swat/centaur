
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def trunc_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive two-digit integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative two-digit integer
    input_dict = {"x": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative integer
    input_dict = {"x": -9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Power of two
    input_dict = {"x": 65536}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative power of two
    input_dict = {"x": -65536}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-digit positive integer
    input_dict = {"x": 123456789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multi-digit negative integer
    input_dict = {"x": -987654321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trunc_3"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_3'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_3'], lib="jax", suffix=3)
