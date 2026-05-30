
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trunc_inputs():
    list_of_inputs = []

    # Input 1: Python scalar integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative scalar integer (np.int32)
    input_dict = {"x": np.int32(-105)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of np.int32 containing positive and negative values
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of np.int64
    x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of np.int16
    x = np.arange(-4, 4, dtype=np.int16).reshape((2, 2, 2))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array of np.int8
    x = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int8)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of np.uint32 (unsigned integers)
    x = np.array([0, 10, 100, 1000, 10000], dtype=np.uint32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with large np.int64 values
    x = np.array([[-9223372036854775807, 9223372036854775807], [0, -1]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of zeros (np.int32)
    x = np.zeros((10,), dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array of np.int16
    x = np.random.randint(-10, 10, size=(1, 2, 2, 2, 2), dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trunc_3"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_3'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_3'], lib="jax", suffix=3)
