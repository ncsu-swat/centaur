
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmin_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays, same shape
    x1 = np.array([1, 3, -2, 5], dtype=np.int32)
    x2 = np.array([2, 1, 4, -10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float32 arrays with broadcasting
    x1 = np.array([[1.5, -2.3, 0.0], [4.1, -0.5, 3.2]], dtype=np.float32)
    x2 = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar arrays
    x1 = np.array(-5.5, dtype=np.float64)
    x2 = np.array(3.2, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D float32 arrays containing NaN, Inf, and -Inf
    x1 = np.array([np.nan, np.inf, -np.inf, 5.0, np.nan], dtype=np.float32)
    x2 = np.array([1.0, np.nan, 2.0, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D int64 arrays with broadcasting
    x1 = np.arange(24, dtype=np.int64).reshape(2, 3, 4) - 12
    x2 = np.ones((1, 3, 1), dtype=np.int64) * 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Boolean arrays
    x1 = np.array([True, False, True, False], dtype=np.bool_)
    x2 = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float64 arrays with mixed shapes for broadcasting
    x1 = np.random.randn(2, 1, 4, 3).astype(np.float64)
    x2 = np.random.randn(1, 5, 1, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Empty arrays (dimension with size 0)
    x1 = np.empty((0, 4), dtype=np.float32)
    x2 = np.empty((1, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D uint8 arrays
    x1 = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    x2 = np.array([[10, 200], [150, 32]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D float16 arrays
    x1 = np.array([-1.0, 0.0, 1.0, 100.0], dtype=np.float16)
    x2 = np.array([0.0, -0.0, 2.0, 50.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmin"] = fmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmin'.")


check_valid('jax.numpy.fmin', generated_inputs['jax.numpy.fmin'], lib="jax", suffix=0)
