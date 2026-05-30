
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arcsinh_inputs():
    list_of_inputs = []

    # Input 1: positive integer (python int)
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer (python int)
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero (python int)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large positive integer (python int)
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: large negative integer (python int)
    input_dict = {"x": -1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy int32 scalar, positive
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy int32 scalar, negative
    input_dict = {"x": np.int32(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy int64 scalar, large value
    input_dict = {"x": np.int64(123456)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy int16 scalar, small value
    input_dict = {"x": np.int16(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy int8 scalar, negative
    input_dict = {"x": np.int8(-3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: numpy int64 scalar, negative large value
    input_dict = {"x": np.int64(-987654)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arcsinh_3"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsinh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsinh_3'.")


check_valid('jax.numpy.arcsinh', generated_inputs['jax.numpy.arcsinh_3'], lib="jax", suffix=3)
