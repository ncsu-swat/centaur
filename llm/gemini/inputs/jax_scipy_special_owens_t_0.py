
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def owens_t_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32
    h = np.array(0.5, dtype=np.float32)
    a = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 2: 1D arrays, float32
    h = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 3: 1D arrays, float64 with negative values
    h = np.array([-2.5, -1.0, -0.1, 0.0], dtype=np.float64)
    a = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 4: 2D arrays, matching shapes
    h = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float32)
    a = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 5: 2D arrays with broadcasting (3, 1) and (1, 4)
    h = np.linspace(-2.0, 2.0, 3, dtype=np.float64)[:, np.newaxis]
    a = np.linspace(-1.0, 1.0, 4, dtype=np.float64)[np.newaxis, :]
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 6: 3D arrays
    h = np.ones((2, 3, 4), dtype=np.float32) * 0.75
    a = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 7: Zeros
    h = np.zeros((5,), dtype=np.float64)
    a = np.zeros((5,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 8: Large positive and negative values
    h = np.array([10.0, -10.0, 5.0], dtype=np.float64)
    a = np.array([100.0, -50.0, 25.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 9: 4D arrays with broadcastable dimensions
    h = np.random.randn(2, 1, 3, 1).astype(np.float32)
    a = np.random.randn(1, 4, 1, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 10: Empty arrays
    h = np.empty((0, 2), dtype=np.float32)
    a = np.empty((0, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    # Input 11: Small near-zero values
    h = np.array([1e-5, -1e-5, 1e-8], dtype=np.float64)
    a = np.array([1e-4, 1e-6, -1e-7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"h": h, "a": a}))

    return list_of_inputs


generated_inputs["jax.scipy.special.owens_t"] = owens_t_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.owens_t' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.owens_t'.")


check_valid('jax.scipy.special.owens_t', generated_inputs['jax.scipy.special.owens_t'], lib="jax", suffix=0)
