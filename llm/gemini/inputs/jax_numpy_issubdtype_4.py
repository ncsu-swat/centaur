
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp

def issubdtype_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"arg1": "int32", "arg2": "int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"arg1": "int32", "arg2": "int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"arg1": "float32", "arg2": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"arg1": "float32", "arg2": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"arg1": "uint8", "arg2": "uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"arg1": "uint32", "arg2": "uint64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"arg1": "complex64", "arg2": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"arg1": "complex128", "arg2": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"arg1": "bool", "arg2": "bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"arg1": "bfloat16", "arg2": "bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {"arg1": "int8", "arg2": "int8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.issubdtype_4"] = issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_4'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_4'], lib="jax", suffix=4)
