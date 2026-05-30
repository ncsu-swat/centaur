
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def promote_types_inputs():
    list_of_inputs = []

    # Input 1: float32 and int32
    input_dict = {
        "a": np.dtype('float32'),
        "b": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 and float32
    input_dict = {
        "a": np.dtype('float64'),
        "b": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 and int32
    input_dict = {
        "a": np.dtype('int64'),
        "b": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 and int16
    input_dict = {
        "a": np.dtype('uint8'),
        "b": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool and int32
    input_dict = {
        "a": np.dtype('bool'),
        "b": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64 and float32
    input_dict = {
        "a": np.dtype('complex64'),
        "b": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128 and float64
    input_dict = {
        "a": np.dtype('complex128'),
        "b": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 and uint16
    input_dict = {
        "a": np.dtype('float16'),
        "b": np.dtype('uint16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 and uint8
    input_dict = {
        "a": np.dtype('int8'),
        "b": np.dtype('uint8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 and int8
    input_dict = {
        "a": np.dtype('float32'),
        "b": np.dtype('int8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: bool and float64
    input_dict = {
        "a": np.dtype('bool'),
        "b": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.promote_types_1"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_1'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_1'], lib="jax", suffix=1)
