
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float arrays, same shape
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer arrays, same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 5], [3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D boolean arrays, same shape
    x = np.array([[[True, False], [False, True]]], dtype=bool)
    y = np.array([[[True, True], [False, False]]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting (3D and 1D)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Float64 with positive and negative values
    x = np.array([-1.5, 0.0, 2.3], dtype=np.float64)
    y = np.array([1.5, -0.0, 2.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D arrays (scalars represented as tensors)
    x = np.array(5, dtype=np.int64)
    y = np.array(6, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Complex arrays
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    y = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Large 2D arrays
    x = np.ones((50, 50), dtype=np.int32)
    y = np.zeros((50, 50), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Mixed dtypes (int32 and float32)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Special float values (NaN, Inf, -Inf)
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    y = np.array([np.nan, np.inf, np.inf, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 4D arrays, same shape
    x = np.random.randint(-10, 10, size=(2, 2, 2, 2))
    y = np.random.randint(-10, 10, size=(2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_1"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_1'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_1'], lib="jax", suffix=1)
