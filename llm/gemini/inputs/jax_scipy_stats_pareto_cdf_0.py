
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32
    list_of_inputs.append(
        {
            "x": np.array(2.5, dtype=np.float32),
            "b": np.array(1.5, dtype=np.float32),
            "loc": np.array(0.0, dtype=np.float32),
            "scale": np.array(1.0, dtype=np.float32),
        }
    )

    # Input 2: 1D arrays of matching shape, float64
    list_of_inputs.append(
        {
            "x": np.array([1.0, 2.0, 3.5, 10.0], dtype=np.float64),
            "b": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.5, -1.0], dtype=np.float64),
            "scale": np.array([1.0, 1.0, 2.0, 0.5], dtype=np.float64),
        }
    )

    # Input 3: 2D arrays of matching shape
    list_of_inputs.append(
        {
            "x": np.array([[1.5, 2.5], [3.0, 4.0]], dtype=np.float32),
            "b": np.array([[2.0, 1.0], [0.8, 3.2]], dtype=np.float32),
            "loc": np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
            "scale": np.array([[1.0, 1.5], [0.5, 2.0]], dtype=np.float32),
        }
    )

    # Input 4: 3D arrays
    list_of_inputs.append(
        {
            "x": np.full((2, 3, 2), 5.0, dtype=np.float32),
            "b": np.full((2, 3, 2), 2.5, dtype=np.float32),
            "loc": np.zeros((2, 3, 2), dtype=np.float32),
            "scale": np.ones((2, 3, 2), dtype=np.float32),
        }
    )

    # Input 5: Broadcasting x (2D) with scalar/1D parameters
    list_of_inputs.append(
        {
            "x": np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
            "b": np.array([1.5], dtype=np.float32),
            "loc": np.array(0.0, dtype=np.float32),
            "scale": np.array(1.0, dtype=np.float32),
        }
    )

    # Input 6: Values of x below support (x < loc + scale), leading to 0 CDF
    list_of_inputs.append(
        {
            "x": np.array([-2.0, 0.0, 0.5, 0.9], dtype=np.float32),
            "b": np.array([1.0, 2.0, 1.5, 3.0], dtype=np.float32),
            "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
    )

    # Input 7: Negative location parameter
    list_of_inputs.append(
        {
            "x": np.array([-3.0, -2.0, -1.0, 0.0], dtype=np.float64),
            "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
            "loc": np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float64),
            "scale": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
        }
    )

    # Input 8: Multi-dimensional broadcasting across dimensions
    list_of_inputs.append(
        {
            "x": np.arange(2, 6, dtype=np.float32).reshape(4, 1),
            "b": np.array([0.5, 1.5, 2.5], dtype=np.float32).reshape(1, 3),
            "loc": np.zeros((1, 1), dtype=np.float32),
            "scale": np.ones((1, 1), dtype=np.float32),
        }
    )

    # Input 9: Integer type inputs
    list_of_inputs.append(
        {
            "x": np.array([2, 4, 8, 16], dtype=np.int32),
            "b": np.array([1, 2, 3, 4], dtype=np.int32),
            "loc": np.array([0, 0, 0, 0], dtype=np.int32),
            "scale": np.array([1, 1, 2, 2], dtype=np.int32),
        }
    )

    # Input 10: Complex broadcasting with varying shapes (2, 1, 3) and (1, 4, 1)
    list_of_inputs.append(
        {
            "x": np.full((2, 1, 3), 10.0, dtype=np.float32),
            "b": np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32),
            "loc": np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32),
            "scale": np.array(2.0, dtype=np.float32),
        }
    )

    # Input 11: Large values of x and b
    list_of_inputs.append(
        {
            "x": np.array([1e4, 1e5, 1e6], dtype=np.float64),
            "b": np.array([10.0, 20.0, 50.0], dtype=np.float64),
            "loc": np.array([0.0, 10.0, 100.0], dtype=np.float64),
            "scale": np.array([100.0, 100.0, 100.0], dtype=np.float64),
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.cdf"] = pareto_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.cdf'.")


check_valid('jax.scipy.stats.pareto.cdf', generated_inputs['jax.scipy.stats.pareto.cdf'], lib="jax", suffix=0)
