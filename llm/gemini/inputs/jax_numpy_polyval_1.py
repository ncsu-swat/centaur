
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays
    p = np.array([2.0, 5.0, 1.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    unroll = 16
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 arrays, 2D x
    p = np.array([1, -2, 3], dtype=np.int32)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    unroll = 8
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 0D x (scalar-like array)
    p = np.array([1.5, -2.5, 0.5, 4.0], dtype=np.float64)
    x = np.array(2.0, dtype=np.float64)
    unroll = 32
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D x, negative values
    p = np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float32)
    x = np.random.uniform(-5, 5, size=(2, 3, 4)).astype(np.float32)
    unroll = 64
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single coefficient (constant polynomial), float32
    p = np.array([5.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    unroll = 4
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex values
    p = np.array([1.0 + 1.0j, -2.0 + 0.0j], dtype=np.complex64)
    x = np.array([0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex64)
    unroll = 16
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int64 arrays, 2D x, unroll=128
    p = np.array([3, 2, 1, 0], dtype=np.int64)
    x = np.array([[1, -1, 0], [2, -2, 5]], dtype=np.int64)
    unroll = 128
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High degree polynomial, float32, unroll=256
    p = np.random.randn(100).astype(np.float32)
    x = np.random.randn(10).astype(np.float32)
    unroll = 256
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float16 arrays
    p = np.array([1.0, -1.0, 2.0], dtype=np.float16)
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float16)
    unroll = 16
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D x array, float64
    p = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    x = np.random.randn(2, 2, 2, 2).astype(np.float64)
    unroll = 16
    input_dict = {"p": p, "x": x, "unroll": unroll}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polyval_1"] = polyval_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyval_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyval_1'.")


check_valid('jax.numpy.polyval', generated_inputs['jax.numpy.polyval_1'], lib="jax", suffix=1)
