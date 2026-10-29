
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def float_power_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays
    x = np.array([3, 1, -5], dtype=np.int32)
    y = np.array([2, 4, -1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D array and 1D broadcast-compatible array
    x = np.array([[2.0, -4.0, 1.0], [-1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalars (represented as 0D numpy arrays)
    x = np.array(4.0, dtype=np.float64)
    y = np.array(0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D arrays with matching shapes
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 0.5], [1.0, -1.0]], [[0.0, 2.0], [-2.0, 1.5]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcast across multiple dimensions (shape (4, 1) and (1, 3))
    x = np.array([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: float64 precision with fractional exponents
    x = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
    y = np.array([0.5, 0.333, -0.5, 1.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Complex numbers
    x = np.array([1.0 + 2.0j, -2.0 + 1.0j, 3.0 - 1.0j], dtype=np.complex64)
    y = np.array([2.0, 0.5, -1.0], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Negative base with integer exponents
    x = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D tensors broadcasting with 1D
    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    y = np.array([0.0, 1.0, 2.0, 3.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Mixed integer type inputs
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([0, 1, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Zeros and ones
    x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 5.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.float_power_1"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.float_power_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.float_power_1'.")


check_valid('jax.numpy.float_power', generated_inputs['jax.numpy.float_power_1'], lib="jax", suffix=1)
