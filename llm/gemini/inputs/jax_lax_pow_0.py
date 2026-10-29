
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pow_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 arrays, identical shapes
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0], [0.0, -2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Scalar arrays (0D)
    x = np.array(5.0, dtype=np.float64)
    y = np.array(3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D array with 0D scalar exponent
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    y = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar base with 1D array exponent
    x = np.array(2.0, dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Integer exponent (should be cast to float)
    x = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    y = np.array([3, 2, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Complex numbers
    x = np.array([1.0 + 2.0j, -2.0 + 1.0j], dtype=np.complex64)
    y = np.array([2.0 + 0.0j, 0.5 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D broadcasting with same rank
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D float64 arrays
    x = np.ones((2, 3, 4), dtype=np.float64) * 2.0
    y = np.full((2, 3, 4), 3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Float16 dtype with positive values
    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    y = np.array([2.0, -1.0, 0.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D broadcasting with same rank
    x = np.array([[[2.0], [3.0]]], dtype=np.float32)
    y = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 12: Complex base with integer exponent
    x = np.array([[1.0 + 1.0j, -1.0 - 1.0j]], dtype=np.complex128)
    y = np.array([[2, 3]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.pow"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pow'.")


check_valid('jax.lax.pow', generated_inputs['jax.lax.pow'], lib="jax", suffix=0)
