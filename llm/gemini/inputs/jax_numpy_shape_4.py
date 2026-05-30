
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: 0D boolean scalar
    input_dict = {"a": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array
    input_dict = {"a": np.array([True, False, True, False], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    input_dict = {"a": np.array([[True, False], [False, True]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array
    input_dict = {"a": np.ones((2, 3, 4), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array with single row
    input_dict = {"a": np.zeros((1, 5), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array
    input_dict = {"a": np.ones((2, 2, 2, 2), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D boolean array
    input_dict = {"a": np.array(False, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D boolean array with size 1 dimensions
    input_dict = {"a": np.zeros((1, 1, 1), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D large boolean array
    input_dict = {"a": np.ones((10,), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D boolean array
    input_dict = {"a": np.zeros((2, 1, 3, 1, 4), dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.shape_4"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_4'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_4'], lib="jax", suffix=4)
