
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, -2.5, 3.5], dtype=np.float32)
    y = np.array([-4.0, 5.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with broadcasting (same rank)
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[10], [20]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar float64 arrays
    x = np.array(3.14159, dtype=np.float64)
    y = np.array(-2.71828, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar int64 with 2D int64 array
    x = np.array(5, dtype=np.int64)
    y = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 arrays with matching shapes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D int32 arrays with broadcasting (same rank)
    x = np.ones((2, 1, 4, 1), dtype=np.int32)
    y = np.full((2, 3, 1, 5), -2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 1D int64 arrays
    x = np.array([100, -200, 300], dtype=np.int64)
    y = np.array([-50, 150, -250], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D float64 arrays with broadcasting (same rank)
    x = np.array([[1.5, -2.5]], dtype=np.float64)
    y = np.array([[0.5, 2.0], [-1.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D float64 array with 0D scalar float64
    x = np.random.randn(3, 2, 5).astype(np.float64)
    y = np.array(10.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 5D float32 arrays with matching shapes
    x = np.zeros((1, 2, 3, 2, 1), dtype=np.float32)
    y = np.ones((1, 2, 3, 2, 1), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D int32 arrays with negative numbers
    x = np.array([[-32, 15], [0, -100]], dtype=np.int32)
    y = np.array([[12, -45], [60, -200]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.add"] = lax_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.add'.")


check_valid('jax.lax.add', generated_inputs['jax.lax.add'], lib="jax", suffix=0)
