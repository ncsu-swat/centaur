
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gammainc_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    a = np.array(2.5, dtype=np.float32)
    x = np.array(1.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 2: 1D arrays of the same shape, float32
    a = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    x = np.array([0.1, 1.5, 3.0, 4.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 3: 2D arrays, float64
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 4: Broadcasting (3, 1) and (1, 4)
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x = np.array([[0.2, 0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 5: x contains zeros (valid non-negative limit)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 6: 3D arrays, float32
    a = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    x = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 7: Very small positive values, float64
    a = np.array([1e-4, 1e-3, 1e-2], dtype=np.float64)
    x = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 8: Large positive values, float64
    a = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    x = np.array([45.0, 110.0, 190.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 9: 1D broadcast against 2D
    a = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    x = np.full((4, 3), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 10: 4D arrays, float32
    a = np.linspace(0.1, 5.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    x = np.linspace(0.0, 10.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 11: 0D `a` broadcasting with multi-dimensional `x`
    a = np.array(3.0, dtype=np.float64)
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.gammainc"] = gammainc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammainc'.")


check_valid('jax.scipy.special.gammainc', generated_inputs['jax.scipy.special.gammainc'], lib="jax", suffix=0)
