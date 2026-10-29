
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D array with small floats close to zero (higher precision test)
    x = np.array([1e-4, 1e-6, 2e-10], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive and negative floats
    x = np.array([2.0, -4.0, 3.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array of integers
    x = np.array([0, 1, -1, 2, -2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float64 array
    x = np.array([[0.0, -1.5, 2.3], [1.0, -0.5, 0.05]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D single-element array
    x = np.array([0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of floats
    x = np.array(
        [[[1.0, -2.0], [3.0, 0.0]], [[-1.0, 2.0], [0.0, -3.0]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with large negative values
    x = np.array([-100.0, -50.0, -20.0, -10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array of boolean values
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D complex numbers array
    x = np.array([1j, -1j, 0.5 + 0.5j, -0.2 - 0.3j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D array of floats
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.expm1_4"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_4'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_4'], lib="jax", suffix=4)
