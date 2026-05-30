
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sign_inputs():
    list_of_inputs = []

    # Input 1: positive integer scalar
    input_dict = {"x": np.array(10, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer scalar
    input_dict = {"x": np.array(-5, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero integer scalar
    input_dict = {"x": np.array(0, dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with positive, negative, and zero
    input_dict = {"x": np.array([-10, 0, 10, -2, 5], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of shape (3, 3) with int8
    input_dict = {"x": np.array([[-1, 2, 0], [4, -5, 6], [0, 0, -9]], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of shape (2, 2, 2) with int16
    input_dict = {"x": np.array([[[1, -2], [3, 0]], [[-5, 6], [0, -8]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of uint8 (unsigned integers, no negatives)
    input_dict = {"x": np.array([0, 5, 255, 128], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy scalar integer
    input_dict = {"x": np.int64(-999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multidimensional 4D array with int32
    x_4d = np.random.randint(-100, 100, size=(2, 2, 3, 3)).astype(np.int32)
    input_dict = {"x": x_4d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of large negative/positive integers
    input_dict = {"x": np.array([-2147483648, 2147483647, 0], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sign_2"] = jax_numpy_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_2'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_2'], lib="jax", suffix=2)
