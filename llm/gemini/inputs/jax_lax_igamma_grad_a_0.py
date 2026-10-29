
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def igamma_grad_a_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    a = np.array(1.5, dtype=np.float32)
    x = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 2: 1D array float32
    a = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    x = np.array([0.1, 1.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 3: 1D array float64
    a = np.array([1.2, 3.4, 5.6], dtype=np.float64)
    x = np.array([0.5, 2.5, 6.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 4: 2D array float32
    a = np.random.uniform(0.5, 5.0, size=(3, 3)).astype(np.float32)
    x = np.random.uniform(0.1, 5.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 5: 2D array float64
    a = np.random.uniform(1.0, 10.0, size=(2, 4)).astype(np.float64)
    x = np.random.uniform(1.0, 10.0, size=(2, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 6: 3D array float32
    a = np.random.uniform(0.1, 3.0, size=(2, 3, 4)).astype(np.float32)
    x = np.random.uniform(0.1, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 7: 2D same shape float32 with singleton dimension
    a = np.array([[0.5], [1.5], [2.5]], dtype=np.float32)
    x = np.array([[0.2], [1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 8: 2D float64 with matching shapes
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 9: 4D array float32
    a = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    x = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 10: 1D array with edge-boundary values
    a = np.array([0.1, 1.0, 10.0, 50.0], dtype=np.float32)
    x = np.array([0.0, 0.001, 15.0, 60.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    return list_of_inputs


generated_inputs["jax.lax.igamma_grad_a"] = igamma_grad_a_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igamma_grad_a' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igamma_grad_a'.")


check_valid('jax.lax.igamma_grad_a', generated_inputs['jax.lax.igamma_grad_a'], lib="jax", suffix=0)
