
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays of same shape
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 0, 3, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float arrays with negative values
    x = np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float32)
    y = np.array([[-1.0, 2.0], [3.0, -4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Broadcasting 1D to 2D
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    y = np.array([1, 5, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar tensors
    x = np.array(5.0, dtype=np.float64)
    y = np.array(5.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float arrays
    x = np.zeros((2, 3, 4), dtype=np.float32)
    y = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Broadcasting with leading singleton dimensions
    x = np.ones((1, 4), dtype=np.int32)
    y = np.ones((3, 1), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Complex numbers
    x = np.array([1.0 + 2.0j, -3.0 + 0.0j], dtype=np.complex64)
    y = np.array([1.0 + 2.0j, 3.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Unsigned integer arrays
    x = np.array([0, 255, 128], dtype=np.uint8)
    y = np.array([0, 254, 128], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D arrays with broadcast compatibility
    x = np.random.randn(2, 1, 4, 5).astype(np.float32)
    y = np.random.randn(1, 3, 4, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty arrays
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.equal_1"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_1'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_1'], lib="jax", suffix=1)
