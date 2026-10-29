
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def isscalar_inputs():
    list_of_inputs = []

    strings = [
        "",
        "hello",
        "123",
        "-45.67",
        "a",
        "True",
        "False",
        "None",
        "complex(1, 2)",
        "special_chars!@#$%^&*()",
        "multi\nline\nstring",
        "unicode_chars_π_θ_λ",
    ]

    for s in strings:
        input_dict = {"element": s}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_5"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_5'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_5'], lib="jax", suffix=5)
