
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: Small positive python integer
    list_of_inputs.append({"x": 3})

    # Input 2: Small negative python integer
    list_of_inputs.append({"x": -2})

    # Input 3: Zero python integer
    list_of_inputs.append({"x": 0})

    # Input 4: Positive np.int32 scalar
    list_of_inputs.append({"x": np.int32(5)})

    # Input 5: Negative np.int64 scalar
    list_of_inputs.append({"x": np.int64(-4)})

    # Input 6: Small positive np.int16 scalar
    list_of_inputs.append({"x": np.int16(8)})

    # Input 7: Negative np.int8 scalar
    list_of_inputs.append({"x": np.int8(-1)})

    # Input 8: Another positive np.int32 scalar
    list_of_inputs.append({"x": np.int32(6)})

    # Input 9: Another negative np.int64 scalar
    list_of_inputs.append({"x": np.int64(-12)})

    # Input 10: Larger positive python integer
    list_of_inputs.append({"x": 10})

    # Input 11: Larger negative python integer
    list_of_inputs.append({"x": -8})

    return list_of_inputs

generated_inputs["jax.numpy.exp2_2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp2_2'.")


check_valid('jax.numpy.exp2', generated_inputs['jax.numpy.exp2_2'], lib="jax", suffix=2)
