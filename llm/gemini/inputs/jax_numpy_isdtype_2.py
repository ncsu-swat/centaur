
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isdtype_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "dtype": np.dtype('int32'),
        "kind": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "dtype": np.dtype('float32'),
        "kind": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "dtype": np.dtype('bool'),
        "kind": np.dtype('bool')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "dtype": np.dtype('int64'),
        "kind": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "dtype": np.dtype('complex64'),
        "kind": np.dtype('complex64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "dtype": np.dtype('uint8'),
        "kind": np.dtype('uint8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "dtype": np.dtype('float64'),
        "kind": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "dtype": np.dtype('int16'),
        "kind": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "dtype": np.dtype('int32'),
        "kind": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "dtype": np.dtype('float32'),
        "kind": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "dtype": np.dtype('complex128'),
        "kind": np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {
        "dtype": np.dtype('float16'),
        "kind": np.dtype('float16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isdtype_2"] = isdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isdtype_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isdtype_2'.")


check_valid('jax.numpy.isdtype', generated_inputs['jax.numpy.isdtype_2'], lib="jax", suffix=2)
