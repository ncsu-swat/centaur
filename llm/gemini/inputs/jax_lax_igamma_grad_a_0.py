
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def igamma_grad_a_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, typical values
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 2: 2D arrays, float32
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 3: Scalar equivalent (0D arrays), float32
    a = np.array(2.0, dtype=np.float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 4: 1D arrays, float64
    a = np.array([0.5, 1.5, 5.0], dtype=np.float64)
    x = np.array([0.1, 2.0, 10.0], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 5: 3D arrays, float32
    a = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    x = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 6: 2D arrays, float64, with larger values
    a = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    x = np.array([[8.0, 18.0], [28.0, 38.0]], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 7: 1D arrays with small fractional values
    a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    x = np.array([0.05, 0.15, 0.25], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 8: 4D arrays, float32
    a = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 9: 1D arrays where x is 0 (edge case, but mathematically valid for x >= 0)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 10: 2D arrays, float64, mixed range
    a = np.array([[0.5, 5.5], [10.5, 100.5]], dtype=np.float64)
    x = np.array([[0.2, 5.2], [10.2, 100.2]], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

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
