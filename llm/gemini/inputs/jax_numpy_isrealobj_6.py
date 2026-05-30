
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Tuple of integers
    input_dict = {"x": (1, 2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple of floats
    input_dict = {"x": (1.0, -2.5, 3.14)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple containing complex number
    input_dict = {"x": (1, 2, 3+4j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of real numpy arrays
    input_dict = {"x": (np.array([1, 2]), np.array([3, 4]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of complex numpy arrays
    input_dict = {"x": (np.array([1, 2j]), np.array([3, 4]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of mixed numeric types (int and float)
    input_dict = {"x": (1, 2.5, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tuple
    input_dict = {"x": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple of booleans
    input_dict = {"x": (True, False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Nested tuple
    input_dict = {"x": ((1, 2), (3, 4))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple of single element (float)
    input_dict = {"x": (1.5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of float64 numpy array
    input_dict = {"x": (np.array([1.0, 2.0], dtype=np.float64),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Tuple of complex128 numpy array
    input_dict = {"x": (np.array([1.0, 2.0j], dtype=np.complex128),)}
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
