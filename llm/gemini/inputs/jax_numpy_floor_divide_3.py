
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive integers, positive float divisor
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = 3.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D array of mixed negative and positive integers, positive float divisor
    x1 = np.array([[-5, -4, -3], [1, 2, 3]], dtype=np.int32)
    x2 = 2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 3D array of floats, negative float divisor
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = -1.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D array of float32, positive float divisor
    x1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Large multidimensional array, small float divisor
    x1 = np.random.randint(-100, 100, size=(5, 5, 5)).astype(np.int64)
    x2 = 10.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 1D array of int64, decimal float divisor
    x1 = np.array([-10, 0, 10, 20], dtype=np.int64)
    x2 = 2.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 4D array of random float64, float divisor
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float64)
    x2 = 0.1
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 0D array (scalar array), float divisor
    x1 = np.array(42, dtype=np.int32)
    x2 = 5.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: 2D array of zeros, float divisor
    x1 = np.zeros((3, 3), dtype=np.float32)
    x2 = 1.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 3D array of mixed large integers, float divisor
    x1 = np.array(
        [[[1000, 2000], [3000, 4000]], [[-1000, -2000], [-3000, -4000]]],
        dtype=np.int64,
    )
    x2 = 150.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_3"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_3'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_3'], lib="jax", suffix=3)
