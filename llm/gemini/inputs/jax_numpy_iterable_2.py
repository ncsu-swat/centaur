
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_iterable_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    list_of_inputs.append({"y": []})

    # Input 2: Single element integer list
    list_of_inputs.append({"y": [0]})

    # Input 3: List of positive integers
    list_of_inputs.append({"y": [1, 2, 3, 4, 5]})

    # Input 4: List of negative integers
    list_of_inputs.append({"y": [-10, -20, -30, -40]})

    # Input 5: List of mixed signed integers
    list_of_inputs.append({"y": [-100, 0, 50, -25, 200]})

    # Input 6: List of floating point numbers
    list_of_inputs.append({"y": [0.0, 1.5, -2.75, 3.14159]})

    # Input 7: Nested 2D list of integers
    list_of_inputs.append({"y": [[1, 2], [3, 4], [5, 6]]})

    # Input 8: Nested 2D list of floats
    list_of_inputs.append({"y": [[-1.5, 2.5], [0.0, -4.2]]})

    # Input 9: Deeply nested 3D list
    list_of_inputs.append({"y": [[[1, 2]], [[3, 4]], [[-1, -2]]]})

    # Input 10: List of boolean values (0 and 1)
    list_of_inputs.append({"y": [True, False, True, True]})

    # Input 11: List with extreme float values
    list_of_inputs.append({"y": [1e-6, -1e6, 2.5e3, 0.0]})

    return list_of_inputs


generated_inputs["jax.numpy.iterable_2"] = jax_numpy_iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_2'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_2'], lib="jax", suffix=2)
