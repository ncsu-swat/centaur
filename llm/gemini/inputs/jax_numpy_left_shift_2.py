
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def left_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, shift by 1
    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    y = 1
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D int32 array, shift by 2
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = 2
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D int32 array, shift by 3
    x = np.arange(8, dtype=np.int32).reshape((2, 2, 2))
    y = 3
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D signed int32 array with negative values, shift by 1
    x = np.array([-5, -4, -3, -2, -1, 0, 1, 2], dtype=np.int32)
    y = 1
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D int32 array, shift by 4
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = 4
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D (scalar) int64 array, shift by 10
    x = np.array(5, dtype=np.int64)
    y = 10
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 4D int32 array, shift by 2
    x = np.ones((2, 2, 2, 2), dtype=np.int32)
    y = 2
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 1D int64 array, shift by 0
    x = np.array([100, 200, 300], dtype=np.int64)
    y = 0
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 3D int64 array, shift by 16
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = 16
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 2D int64 array, shift by 5
    x = np.array([[128, 256], [512, 1024]], dtype=np.int64)
    y = 5
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: 1D int32 array with random integers, shift by 3
    x = np.random.randint(-100, 100, size=(10,)).astype(np.int32)
    y = 3
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.left_shift_2"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_2'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_2'], lib="jax", suffix=2)
