
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    input_dict = {"args": np.array([True, False], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D boolean array
    input_dict = {"args": np.array(True, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NumPy boolean scalar
    input_dict = {"args": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy boolean dtype object
    input_dict = {"args": np.dtype('bool')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array
    input_dict = {"args": np.array([[True, False], [False, True]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another 1D boolean array
    input_dict = {"args": np.array([True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another NumPy boolean scalar
    input_dict = {"args": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D boolean array
    input_dict = {"args": np.array([[[True]]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Python bool type (as a dtype specifier)
    input_dict = {"args": bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array with multiple elements
    input_dict = {"args": np.array([False, False, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.result_type_6"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_6'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_6'], lib="jax", suffix=6)
