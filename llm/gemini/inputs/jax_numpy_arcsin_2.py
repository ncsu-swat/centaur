
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arcsin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive upper boundary
    input_dict = {"x": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative lower boundary
    input_dict = {"x": float(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive common angle value (pi/6)
    input_dict = {"x": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative common angle value (-pi/6)
    input_dict = {"x": float(-0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Approx sqrt(2)/2
    input_dict = {"x": float(0.7071067811865475)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Approx -sqrt(3)/2
    input_dict = {"x": float(-0.8660254037844386)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small positive float
    input_dict = {"x": float(1e-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small negative float
    input_dict = {"x": float(-1e-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Out of domain positive (evaluates to NaN)
    input_dict = {"x": float(2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Out of domain negative (evaluates to NaN)
    input_dict = {"x": float(-2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Arbitrary valid float
    input_dict = {"x": float(0.3333333333333333)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arcsin_2"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsin_2'.")


check_valid('jax.numpy.arcsin', generated_inputs['jax.numpy.arcsin_2'], lib="jax", suffix=2)
