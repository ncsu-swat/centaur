
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fabs_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32, negative
    x = np.int32(-5)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of int32, mixed positive/negative
    x = np.array([-10, 20, -30, 40], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of int64, mixed
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of int16, mixed
    x = np.array([[[-5, 6], [-7, 8]], [[-9, 10], [-11, 12]]], dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of int8, edge values
    x = np.array([-128, 127, 0, -1], dtype=np.int8)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger 2D array of int32
    x = np.random.randint(-100, 100, size=(10, 10), dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array of int64
    x = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar int64, large negative
    x = np.int64(-999999)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of int32, all negative
    x = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array of int32, containing zeros and negative values
    x = np.zeros((3, 3, 3), dtype=np.int32) - 5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fabs_2"] = fabs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fabs_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fabs_2'.")


check_valid('jax.numpy.fabs', generated_inputs['jax.numpy.fabs_2'], lib="jax", suffix=2)
