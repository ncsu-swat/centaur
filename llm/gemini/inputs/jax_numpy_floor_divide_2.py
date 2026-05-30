
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive divisor
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = 3
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D int64 array with negative values, positive divisor
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
    x2 = 3
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 2D float32 array, negative divisor
    x1 = np.array([[6.0, 7.5], [8.0, 9.1]], dtype=np.float32)
    x2 = -2
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 2D int16 array, positive divisor
    x1 = np.array([[12, 15, 18], [21, 24, 27]], dtype=np.int16)
    x2 = 5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 3D int32 array, positive divisor
    x1 = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
    x2 = 2
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 0D array (scalar-like tensor), positive divisor
    x1 = np.array(42, dtype=np.int32)
    x2 = 10
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 1D float64 array, negative divisor
    x1 = np.array([-10.5, -20.0, 35.5], dtype=np.float64)
    x2 = -3
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 1D uint8 array, positive divisor
    x1 = np.array([100, 150, 200, 250], dtype=np.uint8)
    x2 = 12
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: 4D int32 array, positive divisor
    x1 = np.ones((2, 2, 2, 2), dtype=np.int32) * 50
    x2 = 7
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 2D int32 array containing zeros and negatives, positive divisor
    x1 = np.array([[-10, 0, 10], [-20, 0, 20]], dtype=np.int32)
    x2 = 4
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: 1D float32 array, large integer divisor
    x1 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    x2 = 350
    list_of_inputs.append({"x1": x1, "x2": x2})

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
