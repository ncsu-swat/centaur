
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def wrapcauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(1.0, dtype=np.float32)
    c = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 2: 1D arrays of matching shape
    x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], dtype=np.float64)
    c = np.array([0.2, 0.4, 0.5, 0.6, 0.8], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 3: Broadcast 1D x with scalar c
    x = np.linspace(0.1, 2 * np.pi - 0.1, 10, dtype=np.float32)
    c = np.array(0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 4: Broadcast scalar x with 1D c
    x = np.array(np.pi, dtype=np.float64)
    c = np.linspace(0.1, 0.9, 5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 5: 2D arrays
    x = np.random.uniform(0.0, 2 * np.pi, size=(3, 4)).astype(np.float32)
    c = np.random.uniform(0.1, 0.9, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 6: Broadcast 2D arrays (3, 1) and (1, 4)
    x = np.linspace(0.0, np.pi, 3, dtype=np.float64).reshape(3, 1)
    c = np.linspace(0.2, 0.8, 4, dtype=np.float64).reshape(1, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 7: 3D arrays
    x = np.random.uniform(0.0, 2 * np.pi, size=(2, 3, 2)).astype(np.float32)
    c = np.random.uniform(0.05, 0.95, size=(2, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 8: Boundary values within domain [0, 2*pi] and c close to 0 and 1
    x = np.array([0.0, 2 * np.pi], dtype=np.float64)
    c = np.array([0.01, 0.99], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 9: 4D arrays
    x = np.random.uniform(0.5, 1.5, size=(2, 2, 2, 2)).astype(np.float32)
    c = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    # Input 10: Higher-dimensional broadcasting
    x = np.random.uniform(0.0, 2 * np.pi, size=(5, 1, 3)).astype(np.float64)
    c = np.random.uniform(0.1, 0.9, size=(1, 4, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "c": c}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.wrapcauchy.logpdf"] = wrapcauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.wrapcauchy.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.wrapcauchy.logpdf'.")


check_valid('jax.scipy.stats.wrapcauchy.logpdf', generated_inputs['jax.scipy.stats.wrapcauchy.logpdf'], lib="jax", suffix=0)
