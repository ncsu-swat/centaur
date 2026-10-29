
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def subtract_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x = np.array([1.0, 2.5, -3.2, 0.0], dtype=np.float32)
    y = np.array([0.5, -1.0, 2.0, 4.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer arrays with broadcasting
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar arrays
    x = np.array(42.0, dtype=np.float64)
    y = np.array(-7.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float arrays of matching shape
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcasting with singleton dimensions (3D and 1D)
    x = np.ones((4, 1, 5), dtype=np.float32)
    y = np.ones((2, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Complex numbers
    x = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    y = np.array([2.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Signed integer 64-bit types
    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([10, 20, 30], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D tensors broadcasting
    x = np.random.randn(2, 1, 4, 5).astype(np.float32)
    y = np.random.randn(1, 3, 4, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty arrays
    x = np.empty((0, 3), dtype=np.float32)
    y = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D float64 with column broadcasting
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Negative values and zeros with int32
    x = np.array([-5, 0, 5], dtype=np.int32)
    y = np.array([5, 0, -5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.subtract"] = subtract_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.subtract'.")


check_valid('jax.numpy.subtract', generated_inputs['jax.numpy.subtract'], lib="jax", suffix=0)
