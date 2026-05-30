
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input_dict = {"element": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple with a single integer
    input_dict = {"element": (42,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple with floats (including positive and negative values)
    input_dict = {"element": (-1.5, 2.7, 0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple with boolean values
    input_dict = {"element": (True, False, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple with numpy integer types
    input_dict = {"element": (np.int32(10), np.int64(-20))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple with zero-dimensional numpy arrays
    input_dict = {"element": (np.array(5), np.array(-5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple with complex numbers
    input_dict = {"element": (1 + 2j, -3 - 4j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple with numpy float types
    input_dict = {"element": (np.float32(0.5), np.float64(-1.5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple with a None value
    input_dict = {"element": (None,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple with 1D numpy arrays
    input_dict = {"element": (np.array([1, 2]), np.array([3, 4]))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_7"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_7'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_7'], lib="jax", suffix=7)
