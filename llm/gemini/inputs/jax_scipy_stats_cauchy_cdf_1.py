
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_cauchy_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays of float32
    list_of_inputs.append({
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 2: float64 2D arrays
    list_of_inputs.append({
        "x": np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64),
        "loc": np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float64),
        "scale": np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    })

    # Input 3: 0D arrays (scalars)
    list_of_inputs.append({
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(-1.0, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32)
    })

    # Input 4: Large values for x, float64
    list_of_inputs.append({
        "x": np.array([100.0, -100.0, 1000.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64)
    })

    # Input 5: Broadcastable shapes: x (3, 1), loc (1, 3), scale (1, 1)
    list_of_inputs.append({
        "x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        "scale": np.array([[0.5]], dtype=np.float32)
    })

    # Input 6: 3D arrays
    list_of_inputs.append({
        "x": np.random.randn(2, 2, 2).astype(np.float32),
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32)
    })

    # Input 7: Small scale values (close to 0)
    list_of_inputs.append({
        "x": np.array([0.0, 0.1, -0.1], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.01, 0.05, 0.1], dtype=np.float32)
    })

    # Input 8: High scale values
    list_of_inputs.append({
        "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "loc": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "scale": np.array([10.0, 50.0, 100.0], dtype=np.float32)
    })

    # Input 9: Large 1D array of floats
    list_of_inputs.append({
        "x": np.linspace(-10, 10, 20).astype(np.float32),
        "loc": np.zeros(20, dtype=np.float32),
        "scale": np.ones(20, dtype=np.float32)
    })

    # Input 10: Negative locs and diverse scales
    list_of_inputs.append({
        "x": np.array([0.0, 0.0], dtype=np.float64),
        "loc": np.array([-10.0, -5.0], dtype=np.float64),
        "scale": np.array([2.5, 0.5], dtype=np.float64)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.cdf_1"] = generate_cauchy_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.cdf_1'.")


check_valid('jax.scipy.stats.cauchy.cdf', generated_inputs['jax.scipy.stats.cauchy.cdf_1'], lib="jax", suffix=1)
