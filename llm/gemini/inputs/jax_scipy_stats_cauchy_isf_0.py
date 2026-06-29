
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_isf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, valid range of q and positive scale
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 2: 2D arrays, float64
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    loc = np.array([[10.0, -5.0], [0.0, 3.5]], dtype=np.float64)
    scale = np.array([[1.5, 0.5], [2.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 3: Scalar-like 0D arrays
    q = np.array(0.25, dtype=np.float32)
    loc = np.array(-2.0, dtype=np.float32)
    scale = np.array(0.1, dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 4: Broadcastable shapes (q: (3, 1), loc: (1, 3), scale: (1, 1))
    q = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    loc = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.5]], dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 5: Near limits of q (close to 0 and 1)
    q = np.array([1e-5, 0.5, 1.0 - 1e-5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 6: Larger scale values and large loc offsets
    q = np.array([0.3, 0.7], dtype=np.float32)
    loc = np.array([1000.0, -1000.0], dtype=np.float32)
    scale = np.array([50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 7: 3D tensors with randomly sampled values
    q = np.random.uniform(0.01, 0.99, size=(2, 2, 2)).astype(np.float32)
    loc = np.random.normal(0.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 8: Float64 precision with varying extreme parameters
    q = np.array([0.01, 0.99], dtype=np.float64)
    loc = np.array([-10.0, 10.0], dtype=np.float64)
    scale = np.array([0.01, 100.0], dtype=np.float64)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 9: Broadcasting scale to multiple loc and q
    q = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=np.float32)
    loc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    scale = np.array([2.5], dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    # Input 10: 1x5 arrays with linear spaced loc and constant scale
    q = np.full((1, 5), 0.5, dtype=np.float32)
    loc = np.linspace(-5.0, 5.0, 5, dtype=np.float32).reshape(1, 5)
    scale = np.full((1, 5), 1.0, dtype=np.float32)
    list_of_inputs.append({'q': q, 'loc': loc, 'scale': scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.isf"] = jax_scipy_stats_cauchy_isf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.isf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.isf'.")


check_valid('jax.scipy.stats.cauchy.isf', generated_inputs['jax.scipy.stats.cauchy.isf'], lib="jax", suffix=0)
