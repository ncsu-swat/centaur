
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def float_power_inputs():
    list_of_inputs = []

    # Input 1: Basic positive base and positive exponent
    input_dict = {"x": 2.0, "y": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive base and negative exponent
    input_dict = {"x": 4.0, "y": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative base and positive even integer exponent
    input_dict = {"x": -3.0, "y": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative base and positive odd integer exponent
    input_dict = {"x": -2.0, "y": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative base and negative integer exponent
    input_dict = {"x": -5.0, "y": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero base with positive exponent
    input_dict = {"x": 0.0, "y": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Base of 1.0 with large exponent
    input_dict = {"x": 1.0, "y": 100.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive base with fractional exponent
    input_dict = {"x": 9.0, "y": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Fractional base with negative fractional exponent
    input_dict = {"x": 0.25, "y": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float base with zero exponent
    input_dict = {"x": 7.8, "y": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small base with positive exponent
    input_dict = {"x": 1e-4, "y": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Transcendental base with fractional exponent
    input_dict = {"x": 2.718281828459045, "y": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.float_power_2"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.float_power_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.float_power_2'.")


check_valid('jax.numpy.float_power', generated_inputs['jax.numpy.float_power_2'], lib="jax", suffix=2)
