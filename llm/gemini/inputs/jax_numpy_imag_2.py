
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def imag_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"val": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {"val": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer
    input_dict = {"val": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard positive integer
    input_dict = {"val": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate positive integer
    input_dict = {"val": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Moderate negative integer
    input_dict = {"val": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive integer
    input_dict = {"val": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large negative integer
    input_dict = {"val": -1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 32-bit boundary integer
    input_dict = {"val": 2147483647}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative 32-bit boundary integer
    input_dict = {"val": -2147483648}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary odd integer
    input_dict = {"val": 1337}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative odd integer
    input_dict = {"val": -9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.imag_2"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.imag_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.imag_2'.")


check_valid('jax.numpy.imag', generated_inputs['jax.numpy.imag_2'], lib="jax", suffix=2)
