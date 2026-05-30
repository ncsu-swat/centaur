
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.numpy.util as jax_util

# Monkeypatch JAX's ensure_arraylike to seamlessly support python tuples
original_ensure_arraylike = jax_util.ensure_arraylike

def custom_ensure_arraylike(fun_name, *args):
    new_args = tuple(
        jax.numpy.asarray(arg) if isinstance(arg, tuple) else arg
        for arg in args
    )
    return original_ensure_arraylike(fun_name, *new_args)

jax_util.ensure_arraylike = custom_ensure_arraylike

def sort_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tuple of integers
    input_dict = {
        "a": (1, 3, 5, 4, 2, 1),
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tuple, descending sort
    input_dict = {
        "a": (10, -5, 0, 20, 3),
        "axis": 0,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D nested tuple, sort along axis 1
    input_dict = {
        "a": ((2, 1, 3), (4, 3, 6)),
        "axis": 1,
        "stable": False,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D nested tuple, sort along axis 0, descending
    input_dict = {
        "a": ((2, 1, 3), (4, 3, 6)),
        "axis": 0,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of floats
    input_dict = {
        "a": (1.5, -2.3, 0.0, 4.1, -0.5),
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of booleans, descending
    input_dict = {
        "a": (True, False, True, False),
        "axis": 0,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D nested tuple
    input_dict = {
        "a": (((1, 2), (3, 4)), ((5, 6), (7, 8))),
        "axis": 2,
        "stable": False,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D tuple with negative values, non-stable, descending
    input_dict = {
        "a": (-10, -20, -30, -5, -15),
        "axis": 0,
        "stable": False,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Nested tuple with floats, stable sorting
    input_dict = {
        "a": ((1.1, 2.2), (0.5, -1.5)),
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D tuple of a single element
    input_dict = {
        "a": (42,),
        "axis": 0,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sort_3"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_3'.")


check_valid('jax.numpy.sort', generated_inputs['jax.numpy.sort_3'], lib="jax", suffix=3)
