
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    input_dict = {"x1": 10.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative dividend, positive divisor
    input_dict = {"x1": -10.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive dividend, negative divisor
    input_dict = {"x1": 10.0, "x2": -3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative dividend and divisor
    input_dict = {"x1": -10.0, "x2": -3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero dividend
    input_dict = {"x1": 0.0, "x2": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Exact division with fractional numbers
    input_dict = {"x1": 7.5, "x2": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small fractional values
    input_dict = {"x1": 0.1234, "x2": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative fractional values
    input_dict = {"x1": -23.45, "x2": 4.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large float numbers
    input_dict = {"x1": 1e8, "x2": 3.5e3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Dividend smaller than divisor
    input_dict = {"x1": 2.1, "x2": 5.7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Rational approximation
    input_dict = {"x1": 355.0, "x2": 113.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.divmod_7"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_7'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_7'], lib="jax", suffix=7)
