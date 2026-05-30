
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D tuple of ints, float32, 'K', True
    input_dict = {
        "a": (1, 2, 3),
        "dtype": np.float32,
        "order": "K",
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tuple of floats, float64, 'K', False
    input_dict = {
        "a": (1.5, -2.5, 3.14),
        "dtype": np.float64,
        "order": "K",
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tuple of ints, int32, 'K', True
    input_dict = {
        "a": ((1, 2), (3, 4)),
        "dtype": np.int32,
        "order": "K",
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tuple of floats, int64, 'K', False
    input_dict = {
        "a": ((1.1, 2.2), (3.3, 4.4)),
        "dtype": np.int64,
        "order": "K",
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tuple of ints, float32, 'K', False
    input_dict = {
        "a": (((1, 2), (3, 4)), ((5, 6), (7, 8))),
        "dtype": np.float32,
        "order": "K",
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tuple of booleans, bool, 'K', True
    input_dict = {
        "a": (True, False, True),
        "dtype": np.bool_,
        "order": "K",
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D tuple of complex numbers, complex64, 'K', False
    input_dict = {
        "a": (1+2j, 3-4j),
        "dtype": np.complex64,
        "order": "K",
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tuple, float32, 'K', True
    input_dict = {
        "a": (),
        "dtype": np.float32,
        "order": "K",
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D tuple of large ints, int16, 'K', True
    input_dict = {
        "a": (127, -128, 0),
        "dtype": np.int16,
        "order": "K",
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tuple with single elements, int8, 'K', False
    input_dict = {
        "a": (((1,),), ((2,),)),
        "dtype": np.int8,
        "order": "K",
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.asarray_3"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_3'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_3'], lib="jax", suffix=3)
