
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive integer divisor
    x1 = np.array([1.5, 2.7, 3.9, 5.1], dtype=np.float32)
    x2 = 2
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with negative values, negative integer divisor
    x1 = np.array([[-3.5, 4.5], [5.5, -6.5]], dtype=np.float32)
    x2 = -3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, positive integer divisor
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float64)
    x2 = 5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 array, positive integer divisor
    x1 = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    x2 = 4
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, negative integer divisor
    x1 = np.array([[-10, 20], [-30, 40]], dtype=np.int64)
    x2 = -7
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, positive integer divisor
    x1 = np.random.uniform(-100.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = 10
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float16 array, positive integer divisor
    x1 = np.array([-1.2, 2.3, -3.4, 4.5], dtype=np.float16)
    x2 = 3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D float32 array, positive integer divisor
    x1 = np.array(7.5, dtype=np.float32)
    x2 = 2
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with large values, large integer divisor
    x1 = np.array([[1005.0, 2005.0], [3005.0, 4005.0]], dtype=np.float32)
    x2 = 1000
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int32 array with negative values, positive divisor
    x1 = np.array([[[-5, -6], [-7, -8]], [[-9, -10], [-11, -12]]], dtype=np.int32)
    x2 = 3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmod_3"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_3'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_3'], lib="jax", suffix=3)
