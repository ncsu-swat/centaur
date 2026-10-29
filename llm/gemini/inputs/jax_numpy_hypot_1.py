
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hypot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([3.0, 5.0, 8.0], dtype=np.float32)
    x2 = np.array([4.0, 12.0, 15.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D integer arrays with negative values
    x1 = np.array([[3, -2, 5], [9, 1, -4]], dtype=np.int32)
    x2 = np.array([[-5, 6, 8], [2, -7, 0]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Broadcasting 2D and 1D arrays
    x1 = np.array([[3.0, -2.0, 5.0], [9.0, 1.0, -4.0]], dtype=np.float64)
    x2 = np.array([-5.0, 6.0, 8.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalar arrays
    x1 = np.array(3.0, dtype=np.float32)
    x2 = np.array(-4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Broadcasting with singleton dimensions (1, 4) and (3, 1)
    x1 = np.arange(4, dtype=np.float32).reshape(1, 4)
    x2 = np.arange(3, dtype=np.float32).reshape(3, 1)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Float16 precision
    x1 = np.array([[1.5, -2.5], [0.0, 4.0]], dtype=np.float16)
    x2 = np.array([[2.0, 3.5], [-1.0, -0.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 4D arrays with zeros and large values
    x1 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    x2 = np.full((2, 2, 2, 2), 1e8, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Int64 arrays
    x1 = np.array([-10, 0, 10, 100], dtype=np.int64)
    x2 = np.array([20, -5, 0, -100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Boolean arrays
    x1 = np.array([True, False, True, False], dtype=bool)
    x2 = np.array([False, False, True, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Special values (inf and nan)
    x1 = np.array([np.inf, -np.inf, np.nan, 3.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.hypot_1"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hypot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hypot_1'.")


check_valid('jax.numpy.hypot', generated_inputs['jax.numpy.hypot_1'], lib="jax", suffix=1)
