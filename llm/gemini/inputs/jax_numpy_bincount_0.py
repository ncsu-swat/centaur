
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 array with float32 weights
    x = np.array([1, 2, 3, 3, 2, 1], dtype=np.int32)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 0, "length": 5}
    )

    # Input 2: Int64 array with matching int64 weights
    x = np.array([0, 1, 0, 1, 4], dtype=np.int64)
    weights = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 5, "length": 6}
    )

    # Input 3: Array containing negative values (clipped to 0 in JAX)
    x = np.array([-3, -1, 0, 2, 4], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 0, "length": 6}
    )

    # Input 4: Empty arrays
    x = np.array([], dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 0, "length": 0}
    )

    # Input 5: Single element array with minlength equal to length
    x = np.array([3], dtype=np.int32)
    weights = np.array([2.5], dtype=np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 4, "length": 4}
    )

    # Input 6: Values exceeding length (dropped by JAX bincount)
    x = np.array([1, 2, 10, 20, 30], dtype=np.int32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 0, "length": 4}
    )

    # Input 7: uint8 array with float16 weights
    x = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 2, "length": 5}
    )

    # Input 8: Larger array with random integers
    x = np.random.randint(0, 10, size=50).astype(np.int32)
    weights = np.random.randn(50).astype(np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 5, "length": 15}
    )

    # Input 9: All zero array with int32 weights
    x = np.zeros(10, dtype=np.int32)
    weights = np.ones(10, dtype=np.int32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 1, "length": 3}
    )

    # Input 10: Int16 array with negative and positive values
    x = np.array([-5, -2, 0, 1, 3, 5], dtype=np.int16)
    weights = np.array([0.5, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 3, "length": 8}
    )

    # Input 11: All identical positive elements with large length
    x = np.full(8, 2, dtype=np.int64)
    weights = np.full(8, 1.5, dtype=np.float32)
    list_of_inputs.append(
        {"x": x, "weights": weights, "minlength": 0, "length": 10}
    )

    return list_of_inputs


generated_inputs["jax.numpy.bincount"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bincount'.")


check_valid('jax.numpy.bincount', generated_inputs['jax.numpy.bincount'], lib="jax", suffix=0)
