
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def result_type_inputs():
    list_of_inputs = []

    integers = [
        0,
        1,
        -1,
        42,
        -100,
        255,
        -128,
        32767,
        -32768,
        2147483647,
        -2147483648,
        1000000,
    ]

    for val in integers:
        input_dict = {"args": val}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.result_type_4"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_4'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_4'], lib="jax", suffix=4)
