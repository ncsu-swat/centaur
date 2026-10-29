
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def radians_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Common positive angle (180 degrees -> pi)
    input_dict = {"x": 180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Full circle
    input_dict = {"x": 360.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative angle
    input_dict = {"x": -90.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Fractional angle
    input_dict = {"x": 45.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative 180 degrees
    input_dict = {"x": -180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small positive float
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large positive float
    input_dict = {"x": 1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN value
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative fractional angle
    input_dict = {"x": -270.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.radians_4"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_4'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_4'], lib="jax", suffix=4)
