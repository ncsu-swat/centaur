
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def ndim_inputs():
    list_of_inputs = []

    integers = [
        0,
        1,
        -1,
        2,
        -5,
        10,
        -42,
        100,
        -1000,
        65536,
        -2147483648,
        1000000000,
    ]

    for val in integers:
        input_dict = {"a": val}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ndim_2"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_2'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_2'], lib="jax", suffix=2)
