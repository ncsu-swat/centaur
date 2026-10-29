
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 arrays with negative values
    x1 = np.array([[-1.5, -2.5], [-0.5, 3.0]], dtype=np.float64)
    x2 = np.array([[2.0, -1.0], [-4.0, 0.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Broadcasting 1D to 2D
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalar arrays
    x1 = np.array(5.0, dtype=np.float32)
    x2 = np.array(-2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 arrays
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Integer types
    x1 = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    x2 = np.array([[-3, 2, -1], [0, 4, 2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting with singleton dimensions
    x1 = np.random.randn(4, 1, 5).astype(np.float32)
    x2 = np.random.randn(1, 3, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 4D float32 arrays
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large and small values to test numerical stability (avoiding overflow)
    x1 = np.array([1000.0, -1000.0, 50.0], dtype=np.float64)
    x2 = np.array([1000.0, -1000.0, -50.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Special float values like inf and -inf
    x1 = np.array([-np.inf, np.inf, 0.0], dtype=np.float32)
    x2 = np.array([0.0, -np.inf, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Float16 precision
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    x2 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.logaddexp2_1"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_1'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_1'], lib="jax", suffix=1)
