
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with negative and positive values
    x = np.array([-5, -2, 0, 3, 7], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative floats and zero
    x = np.array([-3.5, 0.0, 1.2, -8.9], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 matrix
    x = np.array([[-1.0, 2.5], [-3.2, -4.8]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D complex64 array
    x = np.array([-1.0 + 2.0j, 3.0 - 4.0j, -5.0 + 0.0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex128 array
    x = np.array([[1.0 - 1.0j, -2.0 + 2.0j], [-3.0 - 3.0j, 4.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32 array with a single element
    x = np.array([-42.5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64 array
    x = np.array([-100, 200], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array
    x = np.array([[[-1.0, 2.0], [-3.0, 4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D boolean array
    x = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int16 array with negative values
    x = np.array([[[[-10, 20], [-30, 40]]]], dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float32 array with special floating point values
    x = np.array([-np.inf, np.inf, 0.0, -1.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D int64 array with large integers
    x = np.array([[-1000000, 2000000], [-3000000, 4000000]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.abs_4"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_4'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_4'], lib="jax", suffix=4)
