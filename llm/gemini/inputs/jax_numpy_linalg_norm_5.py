
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 2.0 norm, axis 0, keepdims False
    x = np.array([3.0, 4.0, 12.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 2.0, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, 1.0 norm (L1), axis 0, keepdims True
    x = np.array([-1.5, 2.0, -3.5, 4.0], dtype=np.float64)
    input_dict = {"x": x, "ord": 1.0, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, inf norm, axis 1, keepdims False
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": float("inf"), "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, -inf norm, axis 0, keepdims True
    x = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"x": x, "ord": float("-inf"), "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, 0-norm (count non-zero), axis -1, keepdims False
    x = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, -3.0]], dtype=np.float32)
    input_dict = {"x": x, "ord": 0.0, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, 3.0 norm (p-norm), axis 1, keepdims False
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 3.0, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, 0.5 norm, positive values, axis 2, keepdims True
    x = np.random.uniform(0.1, 5.0, size=(2, 2, 3)).astype(np.float64)
    input_dict = {"x": x, "ord": 0.5, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, 2.0 norm, axis -2, keepdims False
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"x": x, "ord": 2.0, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with complex numbers, 2.0 norm, axis 0, keepdims False
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"x": x, "ord": 2.0, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, 4.0 norm, negative axis, keepdims True
    x = np.array([-2.0, 0.5, -1.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 4.0, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array, inf norm, axis 0, keepdims True
    x = np.random.randn(5, 2, 2).astype(np.float64)
    input_dict = {"x": x, "ord": float("inf"), "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.norm_5"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_5'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_5'], lib="jax", suffix=5)
