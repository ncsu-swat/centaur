
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def wrapcauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D array x, scalar c (float32)
    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)
    c = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays for both, float64
    x = np.linspace(0, 2*np.pi, 10, dtype=np.float64)
    c = np.linspace(0.1, 0.9, 10, dtype=np.float64)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32
    x = np.random.uniform(-np.pi, np.pi, size=(3, 3)).astype(np.float32)
    c = np.random.uniform(0.1, 0.9, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar inputs (0D tensors)
    x = np.array(1.5, dtype=np.float32)
    c = np.array(0.7, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D x, scalar c, float64
    x = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float64)
    c = np.array(0.25, dtype=np.float64)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcastable shapes (x: 3x1, c: 1x4), float32
    x = np.linspace(-np.pi, np.pi, 3).reshape(3, 1).astype(np.float32)
    c = np.array([0.1, 0.3, 0.5, 0.8]).reshape(1, 4).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small c (close to 0)
    x = np.linspace(0, 2*np.pi, 5, dtype=np.float32)
    c = np.array(1e-4, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large c (close to 1)
    x = np.linspace(0, 2*np.pi, 5, dtype=np.float32)
    c = np.array(0.999, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional arrays (4D)
    x = np.random.uniform(-2*np.pi, 2*np.pi, size=(2, 2, 2, 2)).astype(np.float32)
    c = np.random.uniform(0.2, 0.8, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D x, scalar c (float64), negative values in x
    x = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], dtype=np.float64)
    c = np.array(0.6, dtype=np.float64)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
