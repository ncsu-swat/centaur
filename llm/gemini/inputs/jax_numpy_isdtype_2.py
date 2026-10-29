
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isdtype_inputs():
    list_of_inputs = []

    # Input 1: int32 vs int32
    input_dict = {
        "dtype": np.dtype(np.int32),
        "kind": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 vs float32
    input_dict = {
        "dtype": np.dtype(np.float64),
        "kind": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bool vs bool
    input_dict = {
        "dtype": np.dtype(np.bool_),
        "kind": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 vs uint8
    input_dict = {
        "dtype": np.dtype(np.uint8),
        "kind": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128 vs complex64
    input_dict = {
        "dtype": np.dtype(np.complex128),
        "kind": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16 vs int64
    input_dict = {
        "dtype": np.dtype(np.int16),
        "kind": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16 vs float16
    input_dict = {
        "dtype": np.dtype(np.float16),
        "kind": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint32 vs int32
    input_dict = {
        "dtype": np.dtype(np.uint32),
        "kind": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 vs int8
    input_dict = {
        "dtype": np.dtype(np.int8),
        "kind": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex64 vs complex64
    input_dict = {
        "dtype": np.dtype(np.complex64),
        "kind": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint64 vs uint64
    input_dict = {
        "dtype": np.dtype(np.uint64),
        "kind": np.dtype(np.uint64),
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
