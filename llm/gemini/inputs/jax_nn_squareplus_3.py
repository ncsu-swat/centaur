
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squareplus_inputs():
    list_of_inputs = []

    # Input 1: 1D array, scalar tensor b
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    b = np.array(4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 2: 2D array, scalar tensor b
    x = np.random.randn(3, 4).astype(np.float32)
    b = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 3: 3D array, matching shape for b
    x = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.full((2, 3, 5), 4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 4: 4D array, broadcasting 1D b
    x = np.random.randn(2, 4, 3, 5).astype(np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 5: float64 2D array with large negative and positive values
    x = np.array([[-100.0, -10.0, 0.0], [10.0, 50.0, 100.0]], dtype=np.float64)
    b = np.array(4.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 6: 0D scalars for both x and b
    x = np.array(-3.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 7: float16 precision
    x = np.random.randn(4, 4).astype(np.float16)
    b = np.array(4.0, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 8: 5D array with small b
    x = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
    b = np.array(0.01, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 9: 2D array, broadcasting (3, 1) b
    x = np.random.randn(3, 4).astype(np.float32)
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 10: 1D array with zero-valued b
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    b = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 11: Large dimensions with float64
    x = np.random.randn(32, 64).astype(np.float64)
    b = np.array(4.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    return list_of_inputs


generated_inputs["jax.nn.squareplus_3"] = squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_3'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_3'], lib="jax", suffix=3)
