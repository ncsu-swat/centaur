
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: int32 zero
    input_dict = {"arys": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 positive
    input_dict = {"arys": np.int32(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 negative
    input_dict = {"arys": np.int32(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 small positive
    input_dict = {"arys": np.int32(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32 small negative
    input_dict = {"arys": np.int32(-100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 zero
    input_dict = {"arys": np.int64(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 large positive
    input_dict = {"arys": np.int64(999999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 large negative
    input_dict = {"arys": np.int64(-999999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int32 medium positive
    input_dict = {"arys": np.int32(123456)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 medium negative
    input_dict = {"arys": np.int64(-123456)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.atleast_1d_4"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_4'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_4'], lib="jax", suffix=4)
