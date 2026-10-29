
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with positive int
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float array with zero
    x = np.array([[0.0, 1.5], [-2.3, 0.0]], dtype=np.float32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int array with negative int
    x = np.array(
        [[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]], dtype=np.int64
    )
    y = -4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar array with int
    x = np.array(42, dtype=np.int32)
    y = 42
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D boolean array with int
    x = np.array([True, False, True], dtype=np.bool_)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D float64 array with positive int
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    y = 20
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D integer array with zero
    x = np.zeros((2, 2, 2, 2), dtype=np.int32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D uint8 array with large int
    x = np.array([0, 127, 255], dtype=np.uint8)
    y = 255
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D complex array with int
    x = np.array([[1 + 0j, 2 + 1j], [3 + 0j, 4 - 2j]], dtype=np.complex64)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D int16 array with negative int
    x = np.array([-100, 0, 100], dtype=np.int16)
    y = -100
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty array with int
    x = np.array([], dtype=np.int32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_3"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_3'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_3'], lib="jax", suffix=3)
