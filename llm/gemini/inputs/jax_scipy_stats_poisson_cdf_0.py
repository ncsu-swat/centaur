
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_poisson_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with float32
    list_of_inputs.append({
        "k": np.array([0, 1, 2, 3], dtype=np.float32),
        "mu": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    })

    # Input 2: Scalar-like 0D arrays
    list_of_inputs.append({
        "k": np.array(5.0, dtype=np.float32),
        "mu": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32)
    })

    # Input 3: 2D arrays, int32 for k, float64 for mu and loc
    list_of_inputs.append({
        "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "mu": np.array([[2.5, 2.5], [2.5, 2.5]], dtype=np.float64),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    })

    # Input 4: 3D arrays
    list_of_inputs.append({
        "k": np.ones((2, 2, 2), dtype=np.float32) * 3,
        "mu": np.ones((2, 2, 2), dtype=np.float32) * 5.5,
        "loc": np.zeros((2, 2, 2), dtype=np.float32)
    })

    # Input 5: Negative k values
    list_of_inputs.append({
        "k": np.array([-1.0, -0.5, 0.0, 1.5], dtype=np.float32),
        "mu": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    })

    # Input 6: Non-zero positive loc values
    list_of_inputs.append({
        "k": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        "mu": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    })

    # Input 7: Non-zero negative loc values
    list_of_inputs.append({
        "k": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "mu": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    })

    # Input 8: Broadcasting inputs
    list_of_inputs.append({
        "k": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "mu": np.array([1.5, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32)
    })

    # Input 9: High precision float64 arrays
    list_of_inputs.append({
        "k": np.array([10.0, 20.0], dtype=np.float64),
        "mu": np.array([15.5, 18.2], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64)
    })

    # Input 10: Larger values of k and mu
    list_of_inputs.append({
        "k": np.array([100.0, 150.0], dtype=np.float32),
        "mu": np.array([120.0, 130.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.poisson.cdf"] = jax_scipy_stats_poisson_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.poisson.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.poisson.cdf'.")


check_valid('jax.scipy.stats.poisson.cdf', generated_inputs['jax.scipy.stats.poisson.cdf'], lib="jax", suffix=0)
