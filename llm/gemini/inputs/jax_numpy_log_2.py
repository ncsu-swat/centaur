
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_inputs():
    list_of_inputs = []

    # Input 1: Python standard int 1
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python standard int 2
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Python standard int 3
    input_dict = {"x": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Python standard int 10
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Python standard int 100
    input_dict = {"x": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int32 scalar 5
    input_dict = {"x": np.int32(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int32 scalar 15
    input_dict = {"x": np.int32(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.int32 scalar 50
    input_dict = {"x": np.int32(50)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.int64 scalar 8
    input_dict = {"x": np.int64(8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int64 scalar 100
    input_dict = {"x": np.int64(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: np.int64 scalar 1000
    input_dict = {"x": np.int64(1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.log_2"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_2'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_2'], lib="jax", suffix=2)
