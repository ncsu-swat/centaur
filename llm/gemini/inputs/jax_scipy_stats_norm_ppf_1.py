
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_ppf_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 2: Float64 1D arrays
    q = np.array([0.25, 0.75], dtype=np.float64)
    loc = np.array([10.0, -10.0], dtype=np.float64)
    scale = np.array([5.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 3: Scalar-like 0D arrays
    q = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 4: 2D arrays with random values
    q = np.random.uniform(0.1, 0.9, size=(2, 3)).astype(np.float32)
    loc = np.random.normal(0, 1, size=(2, 3)).astype(np.float32)
    scale = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 5: Broadcasting compatible shapes
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array([[1.0], [0.5]], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 6: 3D arrays
    q = np.random.uniform(0.01, 0.99, size=(2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 7: Large values for loc and scale
    q = np.array([0.001, 0.999], dtype=np.float32)
    loc = np.array([1000.0, -1000.0], dtype=np.float32)
    scale = np.array([500.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 8: Small scale values (near zero limit)
    q = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-5, 1e-5], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 9: Extreme quantile values close to 0 and 1
    q = np.array([1e-7, 1 - 1e-7], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 10: Single element 1D arrays
    q = np.array([0.5], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 11: 4D arrays
    q = np.random.uniform(0.1, 0.9, size=(2, 1, 3, 2)).astype(np.float32)
    loc = np.random.normal(0, 1, size=(2, 1, 3, 2)).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.ppf_1"] = jax_scipy_stats_norm_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.ppf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.ppf_1'.")


check_valid('jax.scipy.stats.norm.ppf', generated_inputs['jax.scipy.stats.norm.ppf_1'], lib="jax", suffix=1)
