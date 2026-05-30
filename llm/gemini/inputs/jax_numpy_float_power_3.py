
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def float_power_inputs():
    list_of_inputs = []

    # Input 1: Simple positive integers
    input_dict = {"x": 2, "y": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative base, positive exponent
    input_dict = {"x": -5, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive base, negative exponent
    input_dict = {"x": 10, "y": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero base, positive exponent
    input_dict = {"x": 0, "y": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive base, zero exponent
    input_dict = {"x": 7, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int32 types
    input_dict = {"x": np.int32(3), "y": np.int32(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int64 types with negative base
    input_dict = {"x": np.int64(-2), "y": np.int64(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed numpy integer types
    input_dict = {"x": np.int32(5), "y": np.int64(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large base, small exponent
    input_dict = {"x": 1000, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative base, negative exponent
    input_dict = {"x": -4, "y": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.float_power_3"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.float_power_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.float_power_3'.")


check_valid('jax.numpy.float_power', generated_inputs['jax.numpy.float_power_3'], lib="jax", suffix=3)
