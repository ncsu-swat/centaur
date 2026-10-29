
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Tuple of integers
    input_dict = {"x": (1, 2, 3, -4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple of floats with positive and negative values
    input_dict = {"x": (-2.5, 0.0, 1.25, 3.14159)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of complex numbers
    input_dict = {"x": (1.0 + 2.0j, -3.0 - 4.0j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of booleans
    input_dict = {"x": (True, False, True, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Nested 2D tuple of integers
    input_dict = {"x": ((-1, 2), (3, -4), (0, 5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested 3D tuple of floats
    input_dict = {
        "x": (
            ((-1.0, 2.0), (3.5, -4.2)),
            ((5.1, -6.0), (7.8, 8.9)),
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single-element tuple with zero complex value (imaginary part 0)
    input_dict = {"x": (0.0 + 0.0j,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tuple
    input_dict = {"x": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed numerical tuple promoted to complex
    input_dict = {"x": (1, -2.5, 3.0 + 4.0j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple of numpy fixed-width integer scalars
    input_dict = {"x": (np.int32(-10), np.int32(20), np.int32(0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of numpy float64 scalars
    input_dict = {"x": (np.float64(-3.14), np.float64(2.718), np.float64(1.414))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Tuple of numpy complex128 scalars
    input_dict = {
        "x": (
            np.complex128(-1.5 + 2.5j),
            np.complex128(3.0 - 0.5j),
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isrealobj_6"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_6'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_6'], lib="jax", suffix=6)
