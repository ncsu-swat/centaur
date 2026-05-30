
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []

    # Input 1: Scalar positive integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative integer
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D numpy array of int32 with positive, negative, and zero values
    input_dict = {"x": np.array([1, -2, 3, -4, 0], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D numpy array of int64
    input_dict = {"x": np.array([[10, -20], [30, -40]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D numpy array of int16
    input_dict = {"x": np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D numpy array of uint32 (unsigned integer)
    input_dict = {"x": np.array([10, 20, 30, 40], dtype=np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D numpy array of int8 containing boundary values
    input_dict = {"x": np.array([-128, 0, 127], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scalar integer
    input_dict = {"x": 1234567890}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D numpy array of integers
    input_dict = {"x": np.arange(-8, 8).reshape(2, 2, 2, 2).astype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.floor_2"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_2'.")


check_valid('jax.numpy.floor', generated_inputs['jax.numpy.floor_2'], lib="jax", suffix=2)
