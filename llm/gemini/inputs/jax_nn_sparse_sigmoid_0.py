
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sparse_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array across the range [-2, 2]
    x = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float32 matrix with normal distribution
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar 0D array at boundary
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float64 array
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float16 array with boundary values
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D float32 batch tensor
    x = np.random.randn(2, 4, 8, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with extreme large and small values
    x = np.array([-1e5, -100.0, -1.0001, 0.9999, 100.0, 1e5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D array with all negative values
    x = -np.abs(np.random.randn(4, 5)).astype(np.float32) - 1.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with all positive values exceeding 1
    x = np.abs(np.random.randn(4, 5)).astype(np.float32) + 1.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D float32 tensor
    x = np.random.randn(1, 2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.sparse_sigmoid"] = sparse_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sparse_sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sparse_sigmoid'.")


check_valid('jax.nn.sparse_sigmoid', generated_inputs['jax.nn.sparse_sigmoid'], lib="jax", suffix=0)
