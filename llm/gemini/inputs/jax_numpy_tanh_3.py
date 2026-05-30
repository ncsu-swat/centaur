
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tanh_inputs():
    list_of_inputs = []

    # Input 1: Scalar positive integer (Python int)
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative integer (np.int32)
    input_dict = {"x": np.int32(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of int32
    input_dict = {"x": np.array([-3, -1, 0, 1, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of int64 with negative and positive values
    input_dict = {"x": np.array([0, 1, -1, 2, -2], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of int16
    input_dict = {"x": np.array([[-5, 5], [10, -10]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of int8
    input_dict = {"x": np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array of int32
    input_dict = {"x": np.arange(-4, 4, dtype=np.int32).reshape(2, 2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar zero (np.int64)
    input_dict = {"x": np.int64(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with random integers in range [-100, 100)
    input_dict = {"x": np.random.randint(-100, 100, size=(20,), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of int32
    input_dict = {"x": np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array of uint8 (unsigned integer)
    input_dict = {"x": np.array([0, 5, 10, 20], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tanh_3"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tanh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tanh_3'.")


check_valid('jax.numpy.tanh', generated_inputs['jax.numpy.tanh_3'], lib="jax", suffix=3)
