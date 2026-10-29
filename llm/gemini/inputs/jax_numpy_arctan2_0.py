
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arctan2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x1 = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    x2 = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 arrays with positive and negative values
    x1 = np.array([[-1.0, -2.5], [3.2, 4.0]], dtype=np.float64)
    x2 = np.array([[1.0, -2.5], [-3.2, 0.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Scalar (0D) arrays
    x1 = np.array(-0.5, dtype=np.float32)
    x2 = np.array(0.866, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting 1D with 2D
    x1 = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    x2 = np.array([[1.0], [-2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Integer types (int32)
    x1 = np.array([-5, 0, 5, 10], dtype=np.int32)
    x2 = np.array([5, -5, 0, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Float16 precision
    x1 = np.array([-1.5, 2.5, 0.0], dtype=np.float16)
    x2 = np.array([3.0, -4.0, -1.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 4D tensors with broadcasting
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 1, 4, 5)).astype(np.float32)
    x2 = np.random.uniform(-10.0, 10.0, size=(1, 3, 4, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Non-contiguous / transposed arrays
    x1 = np.random.randn(3, 5).T.astype(np.float64)
    x2 = np.random.randn(5, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Special boundary values (zeros, inf, nan)
    x1 = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    x2 = np.array([0.0, 0.0, 1.0, -1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: int64 types
    x1 = np.array([[-10, 20], [-30, 40]], dtype=np.int64)
    x2 = np.array([[10, -20], [30, -40]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.arctan2"] = arctan2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan2'.")


check_valid('jax.numpy.arctan2', generated_inputs['jax.numpy.arctan2'], lib="jax", suffix=0)
