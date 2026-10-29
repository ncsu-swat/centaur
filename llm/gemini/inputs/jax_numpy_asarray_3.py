
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D tuple of ints
    input_dict = {
        "a": (1, 2, 3, 4, 5),
        "dtype": np.dtype(np.int32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tuple of floats with negative values
    input_dict = {
        "a": (-1.5, 0.0, 2.5, -3.75),
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D nested tuple of ints
    input_dict = {
        "a": ((1, 2, 3), (4, 5, 6)),
        "dtype": np.dtype(np.int64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D nested tuple of floats with negative values
    input_dict = {
        "a": ((-10.0, 20.5), (30.25, -40.75), (50.0, -60.125)),
        "dtype": np.dtype(np.float64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D nested tuple of ints
    input_dict = {
        "a": (((1, -2), (3, -4)), ((5, -6), (7, -8))),
        "dtype": np.dtype(np.int16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tuple of booleans
    input_dict = {
        "a": (True, False, True, True, False),
        "dtype": np.dtype(np.bool_),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D single-element tuple
    input_dict = {
        "a": (-42,),
        "dtype": np.dtype(np.int8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tuple
    input_dict = {
        "a": (),
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D tuple of complex numbers
    input_dict = {
        "a": (1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j),
        "dtype": np.dtype(np.complex64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D nested tuple with uint8 dtype
    input_dict = {
        "a": ((0, 127), (255, 64)),
        "dtype": np.dtype(np.uint8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D nested tuple with float16 dtype
    input_dict = {
        "a": (((1.0,), (2.0,)), ((3.0,), (4.0,))),
        "dtype": np.dtype(np.float16),
        "order": "K",
        "copy": True,
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
