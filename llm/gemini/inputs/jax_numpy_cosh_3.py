
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_cosh_3_inputs():
    list_of_inputs = []

    # Input 1: Scalar positive integer
    x = 5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative integer
    x = -3
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar zero
    x = 0
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of int32
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of int64 with negative and positive values
    x = np.array([[-5, 4], [3, -2]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of int16
    x = np.array([[[1, -1], [2, -2]], [[3, -3], [0, 0]]], dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of int8
    x = np.arange(-10, 11, 2, dtype=np.int8)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array (scalar array) of int32
    x = np.array(-8, dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 4D array of int32 containing zeros and ones
    x = np.zeros((2, 2, 2, 2), dtype=np.int32)
    x[0, 1, 0, 1] = 5
    x[1, 0, 1, 0] = -5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Numpy integer scalar (np.int64)
    x = np.int64(10)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array of uint32 (unsigned integer)
    x = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cosh_3"] = jax_numpy_cosh_3_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_3'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_3'], lib="jax", suffix=3)
