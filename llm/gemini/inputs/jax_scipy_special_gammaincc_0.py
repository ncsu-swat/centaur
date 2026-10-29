
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gammaincc_inputs():
    list_of_inputs = []

    # Input 1: 0-dimensional scalars (float32)
    a = np.array(1.5, dtype=np.float32)
    x = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 2: 1-dimensional arrays (float64)
    a = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float64)
    x = np.array([0.1, 1.0, 2.0, 3.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 3: Broadcastable shapes (column vector and row vector)
    a = np.array([[0.5], [1.5], [3.0]], dtype=np.float32)
    x = np.array([[0.0, 1.0, 2.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 4: 2D matrices of identical shape
    a = np.random.uniform(0.1, 10.0, size=(4, 4)).astype(np.float32)
    x = np.random.uniform(0.0, 10.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 5: x has zeros (boundary case where gammaincc(a, 0) = 1)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.zeros(3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 6: 3D tensors
    a = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    x = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 7: Small positive values for both a and x
    a = np.array([1e-3, 1e-2, 1e-1], dtype=np.float64)
    x = np.array([1e-4, 1e-3, 1e-2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 8: Large positive values for a and x
    a = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    x = np.array([45.0, 105.0, 190.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 9: 4D tensors with broadcasting
    a = np.random.uniform(1.0, 10.0, size=(2, 1, 3, 1)).astype(np.float32)
    x = np.random.uniform(0.5, 10.0, size=(1, 4, 1, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 10: 1D array broadcasted with 0D scalar
    a = np.linspace(0.5, 5.0, 10, dtype=np.float64)
    x = np.array(1.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.gammaincc"] = gammaincc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammaincc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammaincc'.")


check_valid('jax.scipy.special.gammaincc', generated_inputs['jax.scipy.special.gammaincc'], lib="jax", suffix=0)
