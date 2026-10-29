
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def wrapcauchy_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars as tensors, float32
    x = np.array(1.5, dtype=np.float32)
    c = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 2: 1D arrays, float32
    x = np.linspace(0.0, 2 * np.pi, 10, dtype=np.float32)
    c = np.full((10,), 0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 3: 1D arrays, float64, small c
    x = np.linspace(0.1, 6.0, 20, dtype=np.float64)
    c = np.full((20,), 0.05, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 4: 1D arrays, c close to 1
    x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], dtype=np.float32)
    c = np.array([0.9, 0.95, 0.8, 0.85, 0.99], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 5: 2D arrays of same shape
    x = np.random.uniform(0.0, 2 * np.pi, size=(4, 5)).astype(np.float32)
    c = np.random.uniform(0.1, 0.9, size=(4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 6: Broadcasting 2D (x: (3, 1), c: (1, 4))
    x = np.linspace(0.0, 2 * np.pi, 3, dtype=np.float64).reshape(3, 1)
    c = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 7: Broadcasting 1D c across 2D x
    x = np.random.uniform(0.0, 2 * np.pi, size=(3, 6)).astype(np.float32)
    c = np.array([0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 8: 3D arrays
    x = np.random.uniform(0.0, 2 * np.pi, size=(2, 3, 4)).astype(np.float32)
    c = np.full((2, 3, 4), 0.7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 9: 4D arrays with float64
    x = np.random.uniform(0.0, 2 * np.pi, size=(2, 2, 3, 3)).astype(np.float64)
    c = np.random.uniform(0.2, 0.8, size=(2, 2, 3, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 10: Boundary values including exact 0 and 2*pi
    x = np.array([0.0, 2 * np.pi], dtype=np.float64)
    c = np.array([0.5, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 11: Broadcasting 3D with 1D
    x = np.random.uniform(0.0, 2 * np.pi, size=(2, 4, 3)).astype(np.float32)
    c = np.linspace(0.1, 0.9, 3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.wrapcauchy.pdf"] = wrapcauchy_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.wrapcauchy.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.wrapcauchy.pdf'.")


check_valid('jax.scipy.stats.wrapcauchy.pdf', generated_inputs['jax.scipy.stats.wrapcauchy.pdf'], lib="jax", suffix=0)
