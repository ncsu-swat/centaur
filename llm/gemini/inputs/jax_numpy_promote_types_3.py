
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def promote_types_inputs():
    list_of_inputs = []

    # Input 1: int32 dtype and float32 string
    input_dict = {"a": np.dtype('int32'), "b": 'float32'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 dtype and int64 string
    input_dict = {"a": np.dtype('float64'), "b": 'int64'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: uint8 dtype and float16 string
    input_dict = {"a": np.dtype('uint8'), "b": 'float16'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bool dtype and int16 string
    input_dict = {"a": np.dtype('bool'), "b": 'int16'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 dtype and float32 string
    input_dict = {"a": np.dtype('complex64'), "b": 'float32'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8 dtype and uint16 string
    input_dict = {"a": np.dtype('int8'), "b": 'uint16'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 dtype and complex128 string
    input_dict = {"a": np.dtype('float32'), "b": 'complex128'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 dtype and bool string
    input_dict = {"a": np.dtype('int64'), "b": 'bool'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 dtype and float32 string
    input_dict = {"a": np.dtype('float16'), "b": 'float32'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128 dtype and int32 string
    input_dict = {"a": np.dtype('complex128'), "b": 'int32'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.promote_types_3"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_3'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_3'], lib="jax", suffix=3)
