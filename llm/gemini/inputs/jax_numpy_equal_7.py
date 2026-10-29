
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: Zero values
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different positive integers
    input_dict = {"x": 1, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Identical negative integers
    input_dict = {"x": -5, "y": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive and negative with same magnitude
    input_dict = {"x": -100, "y": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Identical positive integers
    input_dict = {"x": 42, "y": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive integers (identical)
    input_dict = {"x": 1000000000, "y": 1000000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive integers (different)
    input_dict = {"x": 123456789, "y": 987654321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer and zero
    input_dict = {"x": -1, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative integers (identical)
    input_dict = {"x": -99999999, "y": -99999999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two distinct powers of two
    input_dict = {"x": 256, "y": 512}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.equal_7"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_7'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_7'], lib="jax", suffix=7)
