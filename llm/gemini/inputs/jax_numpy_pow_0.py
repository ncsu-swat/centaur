
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pow_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D integer arrays with non-negative exponents
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([[3, 2], [1, 0]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar arrays
    x1 = np.array(5.0, dtype=np.float32)
    x2 = np.array(3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting shapes (3, 1) and (1, 4)
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    x2 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Negative base with integer exponents
    x1 = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D float arrays
    x1 = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    x2 = np.full((2, 3, 4), 3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Complex arrays
    x1 = np.array([1.0 + 2.0j, -2.0 + 1.0j], dtype=np.complex64)
    x2 = np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Floating point broadcasting with 1D and 2D arrays
    x1 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([[1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 4D arrays
    x1 = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(0.0, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Array with fractional exponents
    x1 = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64)
    x2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Mixed dimension broadcasting with scalar base
    x1 = np.array([2.0], dtype=np.float32)
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.pow"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pow'.")


check_valid('jax.numpy.pow', generated_inputs['jax.numpy.pow'], lib="jax", suffix=0)
