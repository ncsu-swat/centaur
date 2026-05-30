
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_can_cast_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "from_": np.dtype('int32'),
        "to": "int64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "from_": np.dtype('float32'),
        "to": "float64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "from_": np.dtype('float64'),
        "to": "float32",
        "casting": "unsafe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "from_": np.dtype('int32'),
        "to": "float32",
        "casting": "same_kind"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "from_": np.dtype('complex64'),
        "to": "complex128",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "from_": np.dtype('bool'),
        "to": "int32",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "from_": np.dtype('int16'),
        "to": "int16",
        "casting": "no"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "from_": np.dtype('uint8'),
        "to": "int16",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "from_": np.dtype('float32'),
        "to": "float32",
        "casting": "equiv"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "from_": np.dtype('int32'),
        "to": "float64",
        "casting": "safe"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.can_cast_2"] = jax_numpy_can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_2'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_2'], lib="jax", suffix=2)
