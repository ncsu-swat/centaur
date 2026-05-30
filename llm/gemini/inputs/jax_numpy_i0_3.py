
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_jax_numpy_i0_inputs():
    list_of_inputs = []

    # Input 1: Zero (python int)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer (python int)
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer (python int)
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Medium positive integer (python int)
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Medium negative integer (python int)
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive np.int32 scalar
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative np.int32 scalar
    input_dict = {"x": np.int32(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive np.int64 scalar
    input_dict = {"x": np.int64(20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative np.int64 scalar
    input_dict = {"x": np.int64(-20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger positive integer (python int)
    input_dict = {"x": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger negative integer (python int)
    input_dict = {"x": -50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.i0_3"] = generate_jax_numpy_i0_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.i0_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.i0_3'.")


check_valid('jax.numpy.i0', generated_inputs['jax.numpy.i0_3'], lib="jax", suffix=3)
