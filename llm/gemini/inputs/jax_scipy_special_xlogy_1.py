
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def xlogy_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with standard positive floats
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 0D scalar arrays
    x = np.array(0.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: x is zero where y is positive
    x = np.zeros((3, 3), dtype=np.float64)
    y = np.ones((3, 3), dtype=np.float64) * 5.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D arrays with broadcasting (1, N) and (M, 1)
    x = np.array([[0.5], [1.5], [2.5]], dtype=np.float32)
    y = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Negative x values with positive y
    x = np.array([-1.0, -2.5, 3.0, -4.2], dtype=np.float64)
    y = np.array([2.0, 10.0, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D tensors with random positive floats
    x = np.random.uniform(0.1, 5.0, size=(2, 4, 3)).astype(np.float32)
    y = np.random.uniform(0.1, 10.0, size=(2, 4, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Integer arrays
    x = np.array([[0, 1], [2, 3]], dtype=np.int32)
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D tensors with broadcasting
    x = np.ones((2, 1, 3, 1), dtype=np.float32) * 2.0
    y = np.ones((1, 4, 1, 5), dtype=np.float32) * 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Large values
    x = np.array([1e5, 1e10], dtype=np.float64)
    y = np.array([1e3, 1e6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Array with zeros in x and arbitrary non-negative values in y
    x = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 5.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D float16 arrays
    x = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    y = np.array([2.0, 4.0, 8.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.scipy.special.xlogy_1"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy_1'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy_1'], lib="jax", suffix=1)
