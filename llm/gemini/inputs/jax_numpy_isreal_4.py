
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean
    input_dict = {"x": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array
    input_dict = {"x": np.array([True, False, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    input_dict = {"x": np.array([[True, False], [False, True]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array
    input_dict = {"x": np.random.choice([True, False], size=(3, 3, 3)).astype(bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All True 1D array
    input_dict = {"x": np.ones((10,), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All False 2D array
    input_dict = {"x": np.zeros((4, 4), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty boolean array
    input_dict = {"x": np.array([], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean array (single element)
    input_dict = {"x": np.array([False], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D boolean array
    input_dict = {"x": np.random.choice([True, False], size=(2, 2, 2, 2)).astype(bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D boolean array with specific pattern (identity matrix)
    input_dict = {"x": np.eye(5, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isreal_4"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_4'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_4'], lib="jax", suffix=4)
