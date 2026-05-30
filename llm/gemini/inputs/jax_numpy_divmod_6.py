
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays of int32 (positive values)
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = np.array([3, 4, 7], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D arrays of int32 (negative values)
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    x2 = np.array([3] * 11, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 2D arrays of int64
    x1 = np.array([[15, 25], [35, 45]], dtype=np.int64)
    x2 = np.array([[4, 6], [8, 10]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Broadcasting (2D x 1D) of int32
    x1 = np.array([[12, 13, 14], [15, 16, 17]], dtype=np.int32)
    x2 = np.array([5, 6, 7], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Scalar divisor (int)
    x1 = np.array([100, 200, 300], dtype=np.int32)
    x2 = np.array(7, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 3D arrays of int16
    x1 = np.random.randint(-100, 100, size=(2, 2, 2), dtype=np.int16)
    x2 = np.random.randint(1, 10, size=(2, 2, 2), dtype=np.int16)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Large int64 values
    x1 = np.array([100000000000, 200000000000], dtype=np.int64)
    x2 = np.array([3000000000, 4000000000], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: int8 values with negative and positive elements
    x1 = np.array([-10, -20, 30, 40], dtype=np.int8)
    x2 = np.array([3, -4, 5, -6], dtype=np.int8)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: 4D arrays, int32
    x1 = np.ones((2, 2, 2, 2), dtype=np.int32) * 50
    x2 = np.ones((2, 2, 2, 2), dtype=np.int32) * 11
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Broadcasting with different dimensions (3, 1) and (1, 3)
    x1 = np.array([[10], [20], [30]], dtype=np.int32)
    x2 = np.array([[3, 4, 5]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.divmod_6"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_6'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_6'], lib="jax", suffix=6)
