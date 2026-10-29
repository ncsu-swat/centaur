
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def mod_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays (positive)
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([3, 7, 4, 6], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D integer arrays with negative values
    x1 = np.array([-10, 15, -20, 25], dtype=np.int32)
    x2 = np.array([3, -4, -6, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar arrays
    x1 = np.array(42.5, dtype=np.float32)
    x2 = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D float32 arrays of same shape
    x1 = np.array([[5.5, 7.2], [9.8, 12.1]], dtype=np.float32)
    x2 = np.array([[2.0, 3.1], [4.5, 5.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 2D broadcasting (3, 1) and (1, 4)
    x1 = np.arange(1, 4, dtype=np.int64).reshape(3, 1)
    x2 = np.arange(2, 6, dtype=np.int64).reshape(1, 4)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Float64 arrays with negative numbers
    x1 = np.array([[-3.5, 8.25], [-11.1, 14.7]], dtype=np.float64)
    x2 = np.array([[1.5, -2.5], [3.0, -4.2]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 3D float32 arrays
    x1 = np.random.uniform(5.0, 20.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(1.0, 4.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D int64 arrays
    x1 = np.array([1000000000, 2000000000, 3000000000], dtype=np.int64)
    x2 = np.array([3000, 7000, 4000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Broadcasting 3D with 1D
    x1 = np.random.randint(10, 50, size=(2, 3, 4)).astype(np.int32)
    x2 = np.array([2, 3, 5, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D arrays float32
    x1 = np.random.uniform(10.0, 50.0, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = np.random.uniform(2.0, 6.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 2D float64 with broadcasting scalar
    x1 = np.array([[10.5, 20.5], [30.5, 40.5]], dtype=np.float64)
    x2 = np.array(3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.mod"] = mod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mod'.")


check_valid('jax.numpy.mod', generated_inputs['jax.numpy.mod'], lib="jax", suffix=0)
