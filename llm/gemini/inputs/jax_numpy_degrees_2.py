
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import math

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Pi (180 degrees)
    input_dict = {"x": math.pi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative Pi (-180 degrees)
    input_dict = {"x": -math.pi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Half Pi (90 degrees)
    input_dict = {"x": math.pi / 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative quarter Pi (-45 degrees)
    input_dict = {"x": -math.pi / 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Full circle (2*Pi)
    input_dict = {"x": 2.0 * math.pi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arbitrary positive float (1 radian)
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Arbitrary negative float (-2.5 radians)
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small float close to zero
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive float
    input_dict = {"x": 1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: NaN
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.degrees_2"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_2'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_2'], lib="jax", suffix=2)
