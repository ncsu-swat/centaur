
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: tuple of integers
    input_dict = {"x": (1, 2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tuple of negative integers
    input_dict = {"x": (-5, -10, -15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: tuple of standard floats
    input_dict = {"x": (1.0, 2.5, -3.7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tuple of complex numbers
    input_dict = {"x": (1 + 2j, 3 - 4j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: tuple with mixed float and complex
    input_dict = {"x": (0.0, 1 + 0j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tuple of booleans
    input_dict = {"x": (True, False, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tuple containing a 1D complex numpy array
    input_dict = {"x": (np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: tuple containing a 2D real numpy array
    input_dict = {"x": (np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tuple containing a 3D complex numpy array
    input_dict = {
        "x": (np.ones((2, 3, 4), dtype=np.complex128),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tuple of numpy complex scalars
    input_dict = {
        "x": (np.complex64(1.0 - 2.0j), np.complex64(3.0 + 4.0j))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: tuple of numpy float scalars
    input_dict = {
        "x": (np.float64(-3.14), np.float64(2.718))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: single-element tuple with zero complex value
    input_dict = {"x": (0.0 + 0.0j,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.iscomplexobj_6"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_6'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_6'], lib="jax", suffix=6)
