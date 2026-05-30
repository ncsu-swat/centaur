
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def square_inputs():
    list_of_inputs = []

    # Input 1: Simple positive python integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative python integer
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero python integer
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D numpy array (scalar) of type np.int32
    input_dict = {"x": np.array(42, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D numpy array of np.int64
    input_dict = {"x": np.array([1, -2, 3, -4], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D numpy array of np.int32
    input_dict = {"x": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D numpy array of np.int16
    input_dict = {"x": np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D numpy array of np.int8
    input_dict = {"x": np.array([-128, 0, 127], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive python integer
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D numpy array of np.uint32 (unsigned integer)
    input_dict = {"x": np.array([[10, 20], [30, 40]], dtype=np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D numpy array of np.uint16
    input_dict = {"x": np.array([0, 1, 500], dtype=np.uint16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.square_2"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_2'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_2'], lib="jax", suffix=2)
