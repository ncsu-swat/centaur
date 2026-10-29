
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and positive float
    x1 = np.array([-1.5, 0.0, 2.3, 5.7], dtype=np.float32)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 array and negative float
    x1 = np.array([[-3.0, -1.0], [0.5, -4.2]], dtype=np.float64)
    x2 = -2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 3D int32 array and float zero
    x1 = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    x2 = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D array with NaN and Inf values and finite float
    x1 = np.array([np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    x2 = 3.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 4D float32 array and float
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 2D float32 array and float NaN
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = float("nan")
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D float16 array and positive infinity float
    x1 = np.array([10.0, -20.0, 30.0], dtype=np.float16)
    x2 = float("inf")
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 0D scalar array and negative infinity float
    x1 = np.array(5.0, dtype=np.float32)
    x2 = float("-inf")
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D int64 array and large float
    x1 = np.array([[100, 200], [300, 400]], dtype=np.int64)
    x2 = 250.75
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D float64 array with all negative values and negative float
    x1 = np.array([-10.5, -20.3, -5.1, -100.0], dtype=np.float64)
    x2 = -15.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D float32 array with small values and small float
    x1 = np.full((2, 1, 4), 1e-5, dtype=np.float32)
    x2 = 1e-4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_2"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_2'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_2'], lib="jax", suffix=2)
