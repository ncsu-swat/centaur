
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def can_cast_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "from_": "int32",
        "to": np.dtype(np.int64),
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "from_": "float32",
        "to": np.dtype(np.float64),
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "from_": "float64",
        "to": np.dtype(np.float32),
        "casting": "same_kind",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "from_": "int64",
        "to": np.dtype(np.int32),
        "casting": "unsafe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "from_": "int32",
        "to": np.dtype(np.int32),
        "casting": "no",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "from_": "int32",
        "to": np.dtype(np.int32),
        "casting": "equiv",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "from_": "bool",
        "to": np.dtype(np.int8),
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "from_": "complex64",
        "to": np.dtype(np.complex128),
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "from_": "uint8",
        "to": np.dtype(np.float32),
        "casting": "safe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "from_": "float64",
        "to": np.dtype(np.int32),
        "casting": "unsafe",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "from_": "int16",
        "to": np.dtype(np.uint16),
        "casting": "same_kind",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {
        "from_": "complex128",
        "to": np.dtype(np.float64),
        "casting": "same_kind",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.can_cast_3"] = can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_3'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_3'], lib="jax", suffix=3)
