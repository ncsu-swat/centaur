
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def xlogy_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, testing the x=0, y=0 edge case
    x = np.array(0.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D arrays with standard positive floats
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([0.5, 1.0, np.e, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D arrays with negative values for x and positive values for y
    x = np.array([-2.5, -1.0, 0.0, 1.5], dtype=np.float64)
    y = np.array([2.0, 1.0, 0.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D arrays with identical shape
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcasting (3, 1) with (1, 4)
    x = np.array([[0.0], [1.5], [2.5]], dtype=np.float32)
    y = np.array([[0.1, 1.0, 2.0, 10.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D arrays
    x = np.ones((2, 3, 4), dtype=np.float64)
    y = np.full((2, 3, 4), np.e, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Broadcasting scalar tensor with 2D tensor
    x = np.array(5.0, dtype=np.float32)
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Float16 precision
    x = np.array([0.0, 1.0, 0.5], dtype=np.float16)
    y = np.array([0.0, 2.0, 4.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D tensors
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    y = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Array with small positive values approaching zero
    x = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    y = np.array([1e-8, 1e-6, 1e-4], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Broadcasting 1D array across 3D array
    x = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    y = np.ones((2, 2, 3), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.scipy.special.xlogy"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy'], lib="jax", suffix=0)
