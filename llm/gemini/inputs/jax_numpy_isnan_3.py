
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isnan_inputs():
    list_of_inputs = []

    # Input 1: Simple python integer scalar (positive)
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python integer scalar (negative)
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Python integer scalar (zero)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy int32 scalar
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Numpy int64 scalar (large negative value)
    input_dict = {"x": np.int64(-999999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D Numpy array of int32
    input_dict = {"x": np.array([1, -2, 3, 0], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Numpy array of int64
    input_dict = {"x": np.array([[10, 20], [-30, -40]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D Numpy array of int16
    input_dict = {"x": np.ones((2, 2, 2), dtype=np.int16) * -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D Numpy array of uint8
    input_dict = {"x": np.array([0, 128, 255], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher-dimensional (4D) Numpy array of int32
    input_dict = {"x": np.zeros((1, 2, 3, 2), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isnan_3"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isnan_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isnan_3'.")


check_valid('jax.numpy.isnan', generated_inputs['jax.numpy.isnan_3'], lib="jax", suffix=3)
