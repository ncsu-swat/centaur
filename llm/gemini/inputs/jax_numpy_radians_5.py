
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp


def radians_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive acute angle
    input_dict = {"x": 45}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Right angle
    input_dict = {"x": 90}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Straight angle
    input_dict = {"x": 180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Full turn
    input_dict = {"x": 360}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative angle
    input_dict = {"x": -45}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative straight angle
    input_dict = {"x": -180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative full turn
    input_dict = {"x": -360}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive integer
    input_dict = {"x": 720}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large negative integer
    input_dict = {"x": -1080}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Arbitrary prime integer
    input_dict = {"x": 37}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.radians_5"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_5'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_5'], lib="jax", suffix=5)
