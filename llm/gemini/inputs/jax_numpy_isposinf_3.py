
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def isposinf_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {
        "x": 0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive integer 1
    input_dict = {
        "x": 1,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative integer -1
    input_dict = {
        "x": -1,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: moderate positive integer
    input_dict = {
        "x": 42,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: moderate negative integer
    input_dict = {
        "x": -100,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large positive integer
    input_dict = {
        "x": 1000000,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large negative integer
    input_dict = {
        "x": -999999,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: max 32-bit signed integer
    input_dict = {
        "x": 2**31 - 1,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: min 32-bit signed integer
    input_dict = {
        "x": -(2**31),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small positive prime
    input_dict = {
        "x": 7,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative integer
    input_dict = {
        "x": -42,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: positive integer 10
    input_dict = {
        "x": 10,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isposinf_3"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isposinf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isposinf_3'.")


check_valid('jax.numpy.isposinf', generated_inputs['jax.numpy.isposinf_3'], lib="jax", suffix=3)
