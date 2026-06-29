
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def betainc_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar-like arrays (0-D)
    a = np.array(1.5, dtype=np.float32)
    b = np.array(2.5, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 2: 1D arrays
    a = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    x = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 3: 2D arrays, float64
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float64)
    x = np.array([[0.1, 0.3], [0.7, 0.9]], dtype=np.float64)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 4: Broadcasting shapes
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1, 3)
    b = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)  # (3, 1)
    x = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)  # (3, 3)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 5: Boundary value x = 0.0
    a = np.array([2.0, 3.5], dtype=np.float32)
    b = np.array([1.5, 4.0], dtype=np.float32)
    x = np.array([0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 6: Boundary value x = 1.0
    a = np.array([0.8, 5.0], dtype=np.float32)
    b = np.array([2.2, 0.1], dtype=np.float32)
    x = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 7: Large parameters (a, b > 100)
    a = np.array([100.0, 250.0], dtype=np.float64)
    b = np.array([150.0, 300.0], dtype=np.float64)
    x = np.array([0.4, 0.45], dtype=np.float64)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 8: Very small parameters
    a = np.array([0.01, 0.05], dtype=np.float32)
    b = np.array([0.02, 0.08], dtype=np.float32)
    x = np.array([0.1, 0.9], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 9: 3D arrays
    a = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 5.0
    x = np.full((2, 2, 2), 0.35, dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b, "x": x})

    # Input 10: Mixing very small and very large parameters
    a = np.array([0.1, 500.0], dtype=np.float64)
    b = np.array([500.0, 0.1], dtype=np.float64)
    x = np.array([0.01, 0.99], dtype=np.float64)
    list_of_inputs.append({"a": a, "b": b, "x": x})

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
