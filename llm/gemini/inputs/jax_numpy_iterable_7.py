
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: Standard ASCII string
    input_dict = {"y": "hello"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_dict = {"y": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single character
    input_dict = {"y": "a"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numeric string
    input_dict = {"y": "1234567890"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with spaces and punctuation
    input_dict = {"y": "Hello, World! Testing jax.numpy.iterable."}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiline string
    input_dict = {"y": "line1\nline2\nline3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Special characters and symbols
    input_dict = {"y": "!@#$%^&*()_+-=[]{}|;':\",.<>?/`~"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode string
    input_dict = {"y": "こんにちは世界 / 🌍🚀"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Whitespace and escape characters
    input_dict = {"y": "\t\r\n\v\f"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Long repeated string
    input_dict = {"y": "numpy_and_jax_" * 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Formatted scientific notation string
    input_dict = {"y": "-3.1415926535e-10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.iterable_7"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_7'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_7'], lib="jax", suffix=7)
