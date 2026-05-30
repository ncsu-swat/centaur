
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def issubdtype_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "arg1": np.dtype('int32'),
        "arg2": np.integer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "arg1": np.dtype('float64'),
        "arg2": np.floating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "arg1": np.dtype('complex64'),
        "arg2": np.complexfloating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "arg1": np.dtype('uint8'),
        "arg2": np.unsignedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "arg1": np.dtype('bool'),
        "arg2": np.bool_
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "arg1": np.dtype('int16'),
        "arg2": np.signedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "arg1": np.dtype('float32'),
        "arg2": np.number
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "arg1": np.dtype('int64'),
        "arg2": np.integer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "arg1": np.dtype('float16'),
        "arg2": np.floating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "arg1": np.dtype('complex128'),
        "arg2": np.number
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.issubdtype_1"] = issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_1'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_1'], lib="jax", suffix=1)
