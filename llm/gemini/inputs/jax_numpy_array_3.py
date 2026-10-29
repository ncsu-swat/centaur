
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_array_inputs():
    list_of_inputs = []

    # Input 1: 1D integer tuple, float32 dtype
    input_dict = {
        "object": (1, 2, 3),
        "dtype": np.dtype(np.float32),
        "copy": True,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float tuple with negative numbers, int32 dtype
    input_dict = {
        "object": ((-1.5, 2.0), (3.7, -4.2)),
        "dtype": np.dtype(np.int32),
        "copy": False,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty tuple with higher ndmin
    input_dict = {
        "object": (),
        "dtype": np.dtype(np.float64),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tuple
    input_dict = {
        "object": (True, False, True),
        "dtype": np.dtype(np.bool_),
        "copy": False,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D nested tuple with complex numbers
    input_dict = {
        "object": (((1 + 2j, -3j), (4.0, 0j)), ((5 - 1j, 2), (-3, 4j))),
        "dtype": np.dtype(np.complex64),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tuple expanding dimensions via ndmin
    input_dict = {
        "object": (10, -20, 30),
        "dtype": np.dtype(np.int64),
        "copy": True,
        "order": "K",
        "ndmin": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Nested tuple of mixed numeric types to uint8
    input_dict = {
        "object": ((0, 255), (128, 64)),
        "dtype": np.dtype(np.uint8),
        "copy": False,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element tuple to float16
    input_dict = {
        "object": (3.14159,),
        "dtype": np.dtype(np.float16),
        "copy": True,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D tuple with zeros and negative values
    input_dict = {
        "object": ((-10, 0), (0, -20), (5, -5)),
        "dtype": np.dtype(np.int16),
        "copy": False,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D nested tuple
    input_dict = {
        "object": ((((1.0,),),),),
        "dtype": np.dtype(np.float32),
        "copy": True,
        "order": "K",
        "ndmin": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.array_3"] = jax_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_3'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_3'], lib="jax", suffix=3)
