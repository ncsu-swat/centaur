
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sin_inputs():
    list_of_inputs = []

    # Input 1: 1D array with standard float values
    x = np.array([0.0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of integers including negative values
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of floats
    x = np.array([[0.0, 0.5, 1.0], [-0.5, -1.0, -1.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D single-element array
    x = np.array([np.pi / 3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float64 array with negative and positive multiples of pi
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of floats
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4)) * (np.pi / 12)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex array
    x = np.array([1.0 + 2.0j, -1.0 - 1.0j, 0.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with small and large values
    x = np.array([1e-7, -1e-7, 1e5, -1e5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D array of floats
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sin_4"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_4'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_4'], lib="jax", suffix=4)
