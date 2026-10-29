
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: float32 dtype
    input_dict = {"element": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 dtype
    input_dict = {"element": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 dtype
    input_dict = {"element": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 dtype
    input_dict = {"element": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8 dtype
    input_dict = {"element": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bool dtype
    input_dict = {"element": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64 dtype
    input_dict = {"element": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128 dtype
    input_dict = {"element": np.dtype(np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 dtype
    input_dict = {"element": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 dtype
    input_dict = {"element": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32 dtype
    input_dict = {"element": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int8 dtype
    input_dict = {"element": np.dtype(np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_8"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_8'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_8'], lib="jax", suffix=8)
