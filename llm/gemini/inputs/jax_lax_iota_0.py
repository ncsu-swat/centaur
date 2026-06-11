
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iota_inputs():
    list_of_inputs = []

    # Input 1: int32, size 10
    input_dict = {"dtype": np.int32, "size": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, size 5
    input_dict = {"dtype": np.float32, "size": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64, size 100
    input_dict = {"dtype": np.int64, "size": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, size 1
    input_dict = {"dtype": np.float64, "size": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint32, size 50
    input_dict = {"dtype": np.uint32, "size": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16, size 1000
    input_dict = {"dtype": np.int16, "size": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint16, size 8
    input_dict = {"dtype": np.uint16, "size": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8, size 12
    input_dict = {"dtype": np.int8, "size": 12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8, size 3
    input_dict = {"dtype": np.uint8, "size": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, size 0
    input_dict = {"dtype": np.float32, "size": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int32, size 10000
    input_dict = {"dtype": np.int32, "size": 10000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.iota"] = iota_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.iota' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.iota'.")


check_valid('jax.lax.iota', generated_inputs['jax.lax.iota'], lib="jax", suffix=0)
