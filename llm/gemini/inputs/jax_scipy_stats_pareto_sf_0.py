
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_pareto_sf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32
    input_dict = {
        "x": np.array([1.5, 2.0, 3.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float64, with non-zero loc and scale
    input_dict = {
        "x": np.array([[2.5, 3.5], [4.5, 5.5]], dtype=np.float64),
        "b": np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float64),
        "loc": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        "scale": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalars as 0D arrays
    input_dict = {
        "x": np.array(5.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting inputs, different shapes
    input_dict = {
        "x": np.array([[1.5, 2.5, 3.5]], dtype=np.float32), # (1, 3)
        "b": np.array([[2.0], [3.0]], dtype=np.float32),     # (2, 1)
        "loc": np.array([0.0], dtype=np.float32),            # (1,)
        "scale": np.array([1.0], dtype=np.float32)           # (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large b value (steep decay)
    input_dict = {
        "x": np.array([1.1, 1.2, 1.3], dtype=np.float32),
        "b": np.array([50.0, 50.0, 50.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Values of x below the support (should return 1.0)
    input_dict = {
        "x": np.array([0.5, 0.8, 0.9], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale parameter
    input_dict = {
        "x": np.array([150.0, 200.0], dtype=np.float32),
        "b": np.array([1.2, 1.2], dtype=np.float32),
        "loc": np.array([10.0, 10.0], dtype=np.float32),
        "scale": np.array([100.0, 100.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc parameter
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "b": np.array([2.5, 2.5, 2.5], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([3.0, 3.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D Arrays
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float32) * 4.0,
        "b": np.ones((2, 2, 2), dtype=np.float32) * 1.8,
        "loc": np.ones((2, 2, 2), dtype=np.float32) * 0.5,
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional 4D arrays, mixed scales
    input_dict = {
        "x": np.random.uniform(5.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32),
        "b": np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32),
        "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2, 2), dtype=np.float32) * 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.sf"] = jax_scipy_stats_pareto_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.sf'.")


check_valid('jax.scipy.stats.pareto.sf', generated_inputs['jax.scipy.stats.pareto.sf'], lib="jax", suffix=0)
