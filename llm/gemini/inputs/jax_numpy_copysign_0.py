
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def copysign_inputs():
    list_of_inputs = []

    # Input 1: 1D float arrays with matching shapes
    x1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    x2 = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D arrays with broadcasting (row vector and column vector)
    x1 = np.array([[1.0, -2.0, 3.0]], dtype=np.float64)
    x2 = np.array([[-1.0], [1.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Integer arrays (promotes to float)
    x1 = np.array([5, -3, 0, 8], dtype=np.int32)
    x2 = np.array([-1, 1, -1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalar tensors
    x1 = np.array(-5.5, dtype=np.float32)
    x2 = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D arrays with identical shapes
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Special values (zeros, -0.0, inf, -inf)
    x1 = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    x2 = np.array([-0.0, 0.0, -np.inf, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting a scalar tensor with a 2D tensor
    x1 = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float32)
    x2 = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Broadcasting higher-dimensional tensor with leading 1s
    x1 = np.random.randn(1, 4, 5).astype(np.float64)
    x2 = np.random.randn(3, 1, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 4D tensors with broadcasting
    x1 = np.random.randn(2, 1, 3, 1).astype(np.float32)
    x2 = np.random.randn(1, 4, 1, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Float16 precision
    x1 = np.array([-1.5, 2.5, -3.5], dtype=np.float16)
    x2 = np.array([1.0, -1.0, -1.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Mixed integer types
    x1 = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    x2 = np.array([[1, -1]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.copysign"] = copysign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.copysign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.copysign'.")


check_valid('jax.numpy.copysign', generated_inputs['jax.numpy.copysign'], lib="jax", suffix=0)
