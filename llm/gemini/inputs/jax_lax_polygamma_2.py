
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, digamma (m=0.0)
    m = np.array(0.0, dtype=np.float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 2: 1D arrays, trigamma (m=1.0)
    m = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    x = np.array([0.5, 2.0, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 3: 1D array broadcast with m as single-element 1D array, float64
    m = np.array([2.0], dtype=np.float64)
    x = np.array([1.0, 2.5, 5.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 4: 2D arrays, matching shapes
    m = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    x = np.array([[0.1, 1.2], [2.3, 3.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 5: Broadcast 2D: m of shape (3, 1) and x of shape (1, 4)
    m = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 6: 3D arrays with float64
    m = np.zeros((2, 3, 4), dtype=np.float64)
    x = np.linspace(1.0, 10.0, 24).reshape((2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 7: Higher order polygamma (m=5.0), negative non-integer x
    m = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 8: Large positive values of x
    m = np.array([1.0, 2.0], dtype=np.float32)
    x = np.array([100.0, 500.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 9: 4D arrays
    m = np.ones((2, 2, 2, 2), dtype=np.float32)
    x = np.arange(1, 17, dtype=np.float32).reshape((2, 2, 2, 2)) * 0.5
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    # Input 10: Mixed order m with various fractional x values
    m = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    x = np.array([0.25, 0.75, 1.25, 1.75, 2.25], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m, "x": x}))

    return list_of_inputs


generated_inputs["jax.lax.polygamma_2"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.polygamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.polygamma_2'.")


check_valid('jax.lax.polygamma', generated_inputs['jax.lax.polygamma_2'], lib="jax", suffix=2)
