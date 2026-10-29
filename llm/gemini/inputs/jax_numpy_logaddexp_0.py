
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logaddexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 arrays with negative values
    x1 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    x2 = np.array([[-5.0, -6.0], [-7.0, -8.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar arrays
    x1 = np.array(2.5, dtype=np.float32)
    x2 = np.array(-1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting (3, 1) with (1, 4)
    x1 = np.random.randn(3, 1).astype(np.float32)
    x2 = np.random.randn(1, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Large positive values to check numerical stability against overflow
    x1 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    x2 = np.array([1000.0, 1999.0, 3001.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Very small (large negative) values
    x1 = np.array([-1000.0, -500.0], dtype=np.float64)
    x2 = np.array([-1000.0, -600.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 3D float32 arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Integer arrays
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x2 = np.array([[6, 5, 4], [3, 2, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Float16 arrays
    x1 = np.array([0.5, -0.5, 1.5], dtype=np.float16)
    x2 = np.array([-1.5, 2.0, -2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D arrays with broadcasting (1, 2, 1, 4) and (3, 1, 5, 1)
    x1 = np.random.randn(1, 2, 1, 4).astype(np.float32)
    x2 = np.random.randn(3, 1, 5, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D empty arrays
    x1 = np.array([], dtype=np.float32)
    x2 = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.logaddexp"] = logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp'.")


check_valid('jax.numpy.logaddexp', generated_inputs['jax.numpy.logaddexp'], lib="jax", suffix=0)
