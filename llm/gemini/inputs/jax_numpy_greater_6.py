
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    input_dict = {"x": 5.0, "y": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative and positive float
    input_dict = {"x": -3.5, "y": 1.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Both negative floats
    input_dict = {"x": -10.5, "y": -20.3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zeros (positive and negative zero)
    input_dict = {"x": 0.0, "y": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Equal values
    input_dict = {"x": 42.42, "y": 42.42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Very small positive numbers
    input_dict = {"x": 1e-12, "y": 1e-10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very large positive numbers
    input_dict = {"x": 1e20, "y": 1e19}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Infinity compared to finite float
    input_dict = {"x": float("inf"), "y": 1e300}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative infinity compared to finite float
    input_dict = {"x": float("-inf"), "y": -1e300}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Comparison involving NaN
    input_dict = {"x": float("nan"), "y": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Numbers that are close in value
    input_dict = {"x": 1.0000001, "y": 1.0000002}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Positive vs negative small values
    input_dict = {"x": 1e-5, "y": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.greater_6"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_6'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_6'], lib="jax", suffix=6)
