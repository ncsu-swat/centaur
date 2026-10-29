
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccos_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive upper boundary
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative lower boundary
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive common angle value (cos(pi/3) = 0.5)
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative common angle value (cos(2pi/3) = -0.5)
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive sqrt(2)/2
    input_dict = {"x": 0.7071067811865476}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative sqrt(2)/2
    input_dict = {"x": -0.7071067811865476}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive sqrt(3)/2
    input_dict = {"x": 0.8660254037844386}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative sqrt(3)/2
    input_dict = {"x": -0.8660254037844386}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arbitrary small positive float
    input_dict = {"x": 0.123456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary negative float
    input_dict = {"x": -0.42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Near boundary float
    input_dict = {"x": 0.999999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arccos_2"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccos_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccos_2'.")


check_valid('jax.numpy.arccos', generated_inputs['jax.numpy.arccos_2'], lib="jax", suffix=2)
