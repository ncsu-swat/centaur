
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def promote_types_inputs():
    list_of_inputs = []

    # Input 1: int32 and float32 string
    input_dict = {"a": np.dtype(np.int32), "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float16 and float32 string
    input_dict = {"a": np.dtype(np.float16), "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool and int8 string
    input_dict = {"a": np.dtype(np.bool_), "b": "int8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 and int16 string
    input_dict = {"a": np.dtype(np.uint8), "b": "int16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 and float64 string
    input_dict = {"a": np.dtype(np.complex64), "b": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 and uint64 string
    input_dict = {"a": np.dtype(np.int64), "b": "uint64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 and complex64 string
    input_dict = {"a": np.dtype(np.float32), "b": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8 and bool string
    input_dict = {"a": np.dtype(np.int8), "b": "bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32 and int32 string
    input_dict = {"a": np.dtype(np.uint32), "b": "int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 and float64 string
    input_dict = {"a": np.dtype(np.float64), "b": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: complex128 and int32 string
    input_dict = {"a": np.dtype(np.complex128), "b": "int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: uint16 and uint16 string
    input_dict = {"a": np.dtype(np.uint16), "b": "uint16"}
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
