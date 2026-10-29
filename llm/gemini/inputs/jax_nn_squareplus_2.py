
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squareplus_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive values, standard b
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    b = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 2: 1D array with negative values, small b
    x = np.array([-3.0, -2.0, -1.0, 0.0], dtype=np.float32)
    b = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 3: 2D array, float64
    x = np.random.randn(3, 3).astype(np.float64)
    b = 2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 4: 3D array with mixed values
    x = np.random.uniform(-5.0, 5.0, size=(2, 4, 3)).astype(np.float32)
    b = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 5: 0D scalar array
    x = np.array(1.5, dtype=np.float32)
    b = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 6: 4D tensor, large b
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    b = 16.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 7: Array with zeros, b=0.0
    x = np.zeros((5,), dtype=np.float32)
    b = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 8: Float16 array
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float16)
    b = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 9: Large values in x
    x = np.array([1e3, -1e3, 1e4, -1e4], dtype=np.float32)
    b = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 10: 2D array with large b
    x = np.random.randn(4, 8).astype(np.float64)
    b = 100.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 11: 5D tensor
    x = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    b = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    return list_of_inputs


generated_inputs["jax.nn.squareplus_2"] = squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_2'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_2'], lib="jax", suffix=2)
