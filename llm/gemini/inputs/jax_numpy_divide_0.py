
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divide_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x1 = np.array([10.0, -20.0, 30.0, -40.0], dtype=np.float32)
    x2 = np.array([2.0, 4.0, -5.0, -8.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 arrays with same shape
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[0.5, 1.5], [-2.0, 2.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Broadcasting 1D to 2D
    x1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    x2 = np.array([2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalars (tensors)
    x1 = np.array(15.5, dtype=np.float32)
    x2 = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Integer arrays
    x1 = np.array([-7, 8, 9, -10], dtype=np.int32)
    x2 = np.array([2, -3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D tensors with broadcasting
    x1 = np.ones((2, 3, 4), dtype=np.float32) * 12.0
    x2 = np.full((1, 3, 1), 3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D arrays
    x1 = np.random.uniform(1.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = np.random.uniform(0.5, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Column vector and row vector broadcasting
    x1 = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    x2 = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Int64 arrays
    x1 = np.array([100, -500, 250], dtype=np.int64)
    x2 = np.array([4, 20, -5], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D with 1D scalar-like broadcast
    x1 = np.array([[1.5, -3.5], [7.0, 9.5]], dtype=np.float64)
    x2 = np.array([0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.divide"] = divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divide'.")


check_valid('jax.numpy.divide', generated_inputs['jax.numpy.divide'], lib="jax", suffix=0)
