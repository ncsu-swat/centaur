
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def betainc_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    a = np.array(1.0, dtype=np.float32)
    b = np.array(1.0, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 2: 1D arrays with standard positive values (float32)
    a = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    b = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    x = np.array([0.1, 0.25, 0.5, 0.75], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 3: 2D arrays (float64)
    a = np.full((3, 3), 2.5, dtype=np.float64)
    b = np.full((3, 3), 1.5, dtype=np.float64)
    x = np.linspace(0.1, 0.9, 9, dtype=np.float64).reshape(3, 3)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 4: Broadcasting across 2D shapes
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    b = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    x = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 5: Boundary values for x (0.0 and 1.0)
    a = np.array([0.5, 2.0], dtype=np.float32)
    b = np.array([0.5, 2.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 6: 3D arrays
    a = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 4.0
    x = np.full((2, 2, 2), 0.6, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 7: Single-element 1D arrays (float64)
    a = np.array([0.5], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    x = np.array([0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 8: 0D scalar broadcasted against 2D arrays
    a = np.array(2.0, dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 9: Large parameter values
    a = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    b = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    x = np.array([0.45, 0.5, 0.55], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 10: 4D broadcastable shapes
    a = np.ones((1, 2, 1, 3), dtype=np.float32) * 1.5
    b = np.ones((2, 1, 2, 1), dtype=np.float32) * 2.5
    x = np.full((2, 2, 2, 3), 0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    return list_of_inputs


generated_inputs["jax.lax.betainc"] = betainc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.betainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.betainc'.")


check_valid('jax.lax.betainc', generated_inputs['jax.lax.betainc'], lib="jax", suffix=0)
