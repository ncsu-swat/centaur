
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.5, -2.0, 3.7, 0.0], dtype=np.float32)
    y = np.array([-0.5, 4.2, -1.1, 2.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with broadcasting
    x = np.array([[1, 2, 3]], dtype=np.int32)
    y = np.array([[-1], [0], [5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar arrays
    x = np.array(-5.5, dtype=np.float64)
    y = np.array(12.3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float32 arrays with matching shapes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D array + 1D array broadcasting
    x = np.ones((3, 4, 5), dtype=np.float32)
    y = np.arange(5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Complex arrays
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    y = np.array([0.5 - 1.5j, 2.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Integer arrays with negative numbers (int64)
    x = np.array([-100, 200, -300, 400], dtype=np.int64)
    y = np.array([50, -150, 250, -350], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Unsigned integer arrays (uint8)
    x = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    y = np.array([[5, 15], [25, 35]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D arrays with broadcastable dimension
    x = np.random.randn(2, 1, 4, 1).astype(np.float32)
    y = np.random.randn(1, 3, 1, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty arrays with matching/broadcastable shapes
    x = np.empty((0, 4), dtype=np.float32)
    y = np.empty((1, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.add_1"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_1'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_1'], lib="jax", suffix=1)
