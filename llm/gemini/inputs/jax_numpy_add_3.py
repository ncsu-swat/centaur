
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def add_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive integer
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = 10
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 array, negative integer
    x = np.array([[-1.5, 2.0], [3.5, -4.0]], dtype=np.float32)
    y = -5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar array (float64), zero integer
    x = np.array(3.14159, dtype=np.float64)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int64 array, large integer
    x = np.ones((2, 3, 4), dtype=np.int64)
    y = 1000000
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D float16 array, positive integer
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D float32 array, negative integer
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = -42
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D complex64 array, positive integer
    x = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    y = 7
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D uint8 array, small positive integer
    x = np.array([10, 20, 30], dtype=np.uint8)
    y = 5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D boolean array, integer
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty 1D float32 array, integer
    x = np.array([], dtype=np.float32)
    y = 12
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D int8 array with negative elements, negative integer
    x = np.array([[[-5, -10], [15, 20]]], dtype=np.int8)
    y = -3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.add_3"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_3'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_3'], lib="jax", suffix=3)
