
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: Standard Python integer (positive)
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard Python integer (negative)
    input_dict = {"x": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard Python integer (zero)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy int32 positive integer
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NumPy int32 negative integer
    input_dict = {"x": np.int32(-8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NumPy int64 positive integer
    input_dict = {"x": np.int64(20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NumPy int64 negative integer
    input_dict = {"x": np.int64(-15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NumPy int16 positive integer
    input_dict = {"x": np.int16(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NumPy int8 negative integer
    input_dict = {"x": np.int8(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive Python integer
    input_dict = {"x": 40}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large negative Python integer
    input_dict = {"x": -50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.expm1_3"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_3'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_3'], lib="jax", suffix=3)
