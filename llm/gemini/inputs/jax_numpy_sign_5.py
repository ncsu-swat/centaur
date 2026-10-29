
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sign_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive, negative, and zero values
    x = np.array([0.0, -3.0, 7.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D integer array
    x = np.array([1, -5, 0, 12, -99], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float array
    x = np.array([[1.0, -2.0], [0.0, 4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: Single-element float array
    x = np.array([-1.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: Single-element int array
    x = np.array([42], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D nested array of integers
    x = np.array([[[1, -1], [0, 2]], [[-3, 4], [5, -6]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Small and large magnitude floats
    x = np.array([1e-5, -1e-5, 1e5, -1e5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D float array with negative, zero, positive
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with alternating signs
    x = np.array([[0.5, -0.5, 1.5], [-1.5, 2.5, -2.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D integers with large spread
    x = np.array([100, -200, 300, -400, 500], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Simple 3-element integer array
    x = np.array([-1, 0, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sign_5"] = sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_5'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_5'], lib="jax", suffix=5)
