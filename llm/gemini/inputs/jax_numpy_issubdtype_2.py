
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.numpy as jnp


def issubdtype_inputs():
    list_of_inputs = []

    # Input 1: uint32 against uint32 dtype
    input_dict = {"arg1": "uint32", "arg2": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 against float64 dtype
    input_dict = {"arg1": "float32", "arg2": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 against int64 dtype
    input_dict = {"arg1": "int64", "arg2": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bool against bool dtype
    input_dict = {"arg1": "bool", "arg2": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 against complex64 dtype
    input_dict = {"arg1": "complex64", "arg2": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8 against int16 dtype
    input_dict = {"arg1": "int8", "arg2": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 against float64 dtype
    input_dict = {"arg1": "float64", "arg2": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8 against uint8 dtype
    input_dict = {"arg1": "uint8", "arg2": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 against complex128 dtype
    input_dict = {"arg1": "complex128", "arg2": np.dtype(np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 against float32 dtype
    input_dict = {"arg1": "bfloat16", "arg2": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int32 against int32 dtype
    input_dict = {"arg1": "int32", "arg2": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float16 against float16 dtype
    input_dict = {"arg1": "float16", "arg2": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.issubdtype_2"] = issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_2'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_2'], lib="jax", suffix=2)
