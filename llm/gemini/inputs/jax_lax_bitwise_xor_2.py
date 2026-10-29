
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Zeros
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integers
    input_dict = {"x": 5, "y": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative and positive integer
    input_dict = {"x": -10, "y": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both negative integers
    input_dict = {"x": -15, "y": -30}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Identical integers
    input_dict = {"x": 42, "y": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Powers of two
    input_dict = {"x": 1024, "y": 512}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All bits set (-1) with positive integer
    input_dict = {"x": -1, "y": 255}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero and negative integer
    input_dict = {"x": 0, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Byte bitmasks
    input_dict = {"x": 255, "y": 128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger integers
    input_dict = {"x": 123456, "y": 654321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large negative integers
    input_dict = {"x": -1000000, "y": -2000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.bitwise_xor_2"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_2'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_2'], lib="jax", suffix=2)
