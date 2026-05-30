
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: Positive Python integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative Python integer
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy int32 scalar
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NumPy int64 scalar (negative)
    input_dict = {"x": np.int64(-99999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NumPy int32 scalar (zero)
    input_dict = {"x": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NumPy int64 scalar (positive)
    input_dict = {"x": np.int64(123456789)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large Python integer
    input_dict = {"x": 123456789012345678}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative large Python integer
    input_dict = {"x": -1234567890}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NumPy int32 scalar (minimum value)
    input_dict = {"x": np.int32(-2147483648)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NumPy int64 scalar (maximum value)
    input_dict = {"x": np.int64(9223372036854775807)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_3"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_3'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_3'], lib="jax", suffix=3)
