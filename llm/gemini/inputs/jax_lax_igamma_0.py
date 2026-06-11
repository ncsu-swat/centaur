
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def igamma_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays
    a = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    x = np.random.uniform(0.0, 10.0, size=(3, 3)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 arrays
    a = np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float64)
    x = np.random.uniform(0.0, 5.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars as tensors)
    a = np.array(2.5, dtype=np.float32)
    x = np.array(1.2, dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 arrays with larger values
    a = np.random.uniform(10.0, 100.0, size=(5,)).astype(np.float64)
    x = np.random.uniform(10.0, 100.0, size=(5,)).astype(np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x has zero values (valid for x >= 0)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 arrays
    a = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    x = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive values
    a = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
    x = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcastable shapes
    a = np.random.uniform(0.5, 5.0, size=(3, 1)).astype(np.float32)
    x = np.random.uniform(0.5, 5.0, size=(1, 3)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimensions and larger sizes
    a = np.random.uniform(1.0, 5.0, size=(10, 10)).astype(np.float32)
    x = np.random.uniform(1.0, 5.0, size=(10, 10)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.igamma"] = igamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igamma'.")


check_valid('jax.lax.igamma', generated_inputs['jax.lax.igamma'], lib="jax", suffix=0)
