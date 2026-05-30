
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.numpy.arccosh to support Python list inputs directly
_orig_arccosh = jax.numpy.arccosh

def _patched_arccosh(x, *args, **kwargs):
    if isinstance(x, list):
        x = jax.numpy.array(x)
    return _orig_arccosh(x, *args, **kwargs)

jax.numpy.arccosh = _patched_arccosh

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Simple list of positive integers >= 1
    input_dict = {"x": [1, 2, 5, 10]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of floats >= 1.0
    input_dict = {"x": [1.0, 1.5, 2.3, 100.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List containing values < 1 (which yield nan for real float)
    input_dict = {"x": [-5.0, -1.0, 0.0, 0.5, 1.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D list (matrix) of floats >= 1.0
    input_dict = {"x": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of complex numbers
    input_dict = {"x": [complex(1, 2), complex(-5, 0), complex(0, 1)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D list of complex numbers
    input_dict = {"x": [[complex(1, 1), complex(2, -2)], [complex(-3, 3), complex(4, 0)]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D list of floats
    input_dict = {"x": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List with very large float values
    input_dict = {"x": [1e5, 1e10, 1e20]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List containing float infinity
    input_dict = {"x": [1.0, float('inf')]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single-element list
    input_dict = {"x": [1.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Mixed int and float list
    input_dict = {"x": [1, 2.5, 3, 4.2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arccosh_4"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_4'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_4'], lib="jax", suffix=4)
