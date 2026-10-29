
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays with positive and negative values
    x1 = np.array([1, 3, -7, 8], dtype=np.int32)
    x2 = np.array([-1, 4, 6, -9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float arrays of same shape
    x1 = np.array([[1.5, -2.3], [0.0, 4.2]], dtype=np.float32)
    x2 = np.array([[-0.5, 3.1], [-1.2, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Arrays with NaN and Inf values
    x1 = np.array([np.inf, 6.0, -np.inf, np.nan], dtype=np.float32)
    x2 = np.array([3.0, np.nan, np.nan, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting 1D to 2D
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([[5.0], [0.0], [-2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D arrays with float64
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar arrays
    x1 = np.array(5, dtype=np.int64)
    x2 = np.array(-3, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting 0D to 3D
    x1 = np.array(0.0, dtype=np.float32)
    x2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Boolean tensors
    x1 = np.array([[True, False], [False, True]], dtype=np.bool_)
    x2 = np.array([[False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Unsigned integer tensors
    x1 = np.array([0, 255, 128], dtype=np.uint8)
    x2 = np.array([1, 100, 200], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D tensors with broadcasting
    x1 = np.random.randn(1, 3, 1, 5).astype(np.float32)
    x2 = np.random.randn(2, 1, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Empty/Zero-sized arrays
    x1 = np.array([], dtype=np.float32)
    x2 = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs

generated_inputs["jax.numpy.fmax_1"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_1'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_1'], lib="jax", suffix=1)
