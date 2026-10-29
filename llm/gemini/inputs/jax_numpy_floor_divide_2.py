
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive divisor
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D int64 array, negative divisor
    x1 = np.array([10, -20, 30, -40, 50], dtype=np.int64)
    x2 = -4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float32 array, positive divisor
    x1 = np.array([[1.5, 2.7, 3.2], [4.9, 5.1, 6.8]], dtype=np.float32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D float64 array, negative divisor
    x1 = np.array([[-10.5, 0.0], [12.3, -4.7]], dtype=np.float64)
    x2 = -3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D int32 array
    x1 = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array
    x1 = np.array(42, dtype=np.int32)
    x2 = 7
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array
    x1 = np.random.uniform(-100, 100, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = 10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: uint8 array
    x1 = np.array([0, 15, 128, 255], dtype=np.uint8)
    x2 = 16
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large 1D array with divisor 1
    x1 = np.linspace(-50, 50, 100, dtype=np.float32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D int16 array with negative values and divisor -1
    x1 = np.array([[-32, 16], [0, -1]], dtype=np.int16)
    x2 = -1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D array containing zeros
    x1 = np.array([0, 0, 0, 0], dtype=np.int32)
    x2 = 8
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_2"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_2'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_2'], lib="jax", suffix=2)
