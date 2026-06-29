
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays, float32
    list_of_inputs.append({
        'x': np.array([0.0, 1.0, 2.0], dtype=np.float32),
        'loc': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 2: 2D arrays with negative values, float64
    list_of_inputs.append({
        'x': np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float64),
        'loc': np.array([[-0.5, 0.5], [1.5, 2.5]], dtype=np.float64),
        'scale': np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    })

    # Input 3: 0D arrays (scalars represented as numpy arrays)
    list_of_inputs.append({
        'x': np.array(0.5, dtype=np.float32),
        'loc': np.array(-1.2, dtype=np.float32),
        'scale': np.array(2.3, dtype=np.float32)
    })

    # Input 4: Large scale values
    list_of_inputs.append({
        'x': np.random.randn(5).astype(np.float32),
        'loc': np.random.randn(5).astype(np.float32),
        'scale': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    })

    # Input 5: Small scale values
    list_of_inputs.append({
        'x': np.random.randn(2, 3).astype(np.float32),
        'loc': np.random.randn(2, 3).astype(np.float32),
        'scale': np.array([[0.01, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32)
    })

    # Input 6: Broadcasting with different dimensions
    list_of_inputs.append({
        'x': np.random.randn(3, 4).astype(np.float32),
        'loc': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        'scale': np.array(1.5, dtype=np.float32)
    })

    # Input 7: 3D arrays
    list_of_inputs.append({
        'x': np.random.randn(2, 2, 2).astype(np.float32),
        'loc': np.random.randn(2, 2, 2).astype(np.float32),
        'scale': np.abs(np.random.randn(2, 2, 2)).astype(np.float32) + 0.1
    })

    # Input 8: 4D arrays
    list_of_inputs.append({
        'x': np.random.randn(2, 2, 2, 2).astype(np.float64),
        'loc': np.random.randn(2, 2, 2, 2).astype(np.float64),
        'scale': np.ones((2, 2, 2, 2), dtype=np.float64) * 2.5
    })

    # Input 9: Large negative x values where CDF should be near 0
    list_of_inputs.append({
        'x': np.array([-10.0, -20.0, -30.0], dtype=np.float32),
        'loc': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 10: Large positive x values where CDF should be near 1
    list_of_inputs.append({
        'x': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'loc': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.cdf"] = gumbel_l_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.cdf'.")


check_valid('jax.scipy.stats.gumbel_l.cdf', generated_inputs['jax.scipy.stats.gumbel_l.cdf'], lib="jax", suffix=0)
