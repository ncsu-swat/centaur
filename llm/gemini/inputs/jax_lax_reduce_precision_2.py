
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def reduce_precision_inputs():
    list_of_inputs = []

    # Input 1: positive float with bfloat16-like precision
    input_dict = {
        "operand": 3.1415926535,
        "exponent_bits": 8,
        "mantissa_bits": 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float with float16-like precision
    input_dict = {
        "operand": -2.7182818284,
        "exponent_bits": 5,
        "mantissa_bits": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero with very low precision
    input_dict = {
        "operand": 0.0,
        "exponent_bits": 3,
        "mantissa_bits": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large float with intermediate precision
    input_dict = {
        "operand": 12345.6789,
        "exponent_bits": 8,
        "mantissa_bits": 12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small negative float
    input_dict = {
        "operand": -0.00001234,
        "exponent_bits": 6,
        "mantissa_bits": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: standard float32-like precision on a medium float
    input_dict = {
        "operand": 100.5,
        "exponent_bits": 8,
        "mantissa_bits": 23
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: extremely small exponent and mantissa
    input_dict = {
        "operand": 1.5,
        "exponent_bits": 2,
        "mantissa_bits": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative float, custom precision
    input_dict = {
        "operand": -987.654,
        "exponent_bits": 7,
        "mantissa_bits": 15
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: very large float
    input_dict = {
        "operand": 1e6,
        "exponent_bits": 8,
        "mantissa_bits": 16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small positive float
    input_dict = {
        "operand": 0.001953125,
        "exponent_bits": 4,
        "mantissa_bits": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_precision_2"] = reduce_precision_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_precision_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_precision_2'.")


check_valid('jax.lax.reduce_precision', generated_inputs['jax.lax.reduce_precision_2'], lib="jax", suffix=2)
