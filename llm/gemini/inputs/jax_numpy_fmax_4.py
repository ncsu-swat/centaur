
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive integer
    x1 = np.array([1.0, 5.0, -2.0, 10.0], dtype=np.float32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D integer array with zero
    x1 = np.array([-5, 0, 5, 10], dtype=np.int32)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float array with negative integer
    x1 = np.array([[-10.5, -2.0], [3.5, -8.1]], dtype=np.float32)
    x2 = -5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D array with inf and nan values
    x1 = np.array([[np.inf, np.nan], [-np.inf, 2.0]], dtype=np.float64)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D int64 array with positive integer
    x1 = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array
    x1 = np.array(-3.14, dtype=np.float32)
    x2 = -2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D unsigned integer array
    x1 = np.array([0, 10, 255, 128], dtype=np.uint8)
    x2 = 50
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float16 array with large integer
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    x2 = 100
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D bool array
    x1 = np.array([True, False, True, False], dtype=np.bool_)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D float64 array with all NaNs
    x1 = np.full((2, 3, 2), np.nan, dtype=np.float64)
    x2 = -10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_4"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_4'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_4'], lib="jax", suffix=4)
