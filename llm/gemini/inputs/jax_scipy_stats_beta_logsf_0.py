
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    list_of_inputs.append(
        {
            "x": np.array(0.5, dtype=np.float32),
            "a": np.array(2.0, dtype=np.float32),
            "b": np.array(2.0, dtype=np.float32),
            "loc": np.array(0.0, dtype=np.float32),
            "scale": np.array(1.0, dtype=np.float32),
        }
    )

    # Input 2: 1D arrays, standard [0, 1] support
    list_of_inputs.append(
        {
            "x": np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64),
            "a": np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            "b": np.array([3.0, 2.5, 2.0, 1.5, 1.0], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        }
    )

    # Input 3: Broadcasted shapes (2D x and 1D parameters)
    list_of_inputs.append(
        {
            "x": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
            "a": np.array([0.5, 1.5], dtype=np.float32),
            "b": np.array([1.5, 0.5], dtype=np.float32),
            "loc": np.array([0.0], dtype=np.float32),
            "scale": np.array([1.0], dtype=np.float32),
        }
    )

    # Input 4: Non-zero loc and non-unit scale, float32
    list_of_inputs.append(
        {
            "x": np.array([2.5, 3.0, 3.5], dtype=np.float32),
            "a": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "b": np.array([4.0, 3.0, 2.0], dtype=np.float32),
            "loc": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        }
    )

    # Input 5: Negative loc parameter
    list_of_inputs.append(
        {
            "x": np.array([-1.5, -1.0, -0.5], dtype=np.float64),
            "a": np.array([1.2, 2.2, 3.2], dtype=np.float64),
            "b": np.array([2.1, 1.8, 0.9], dtype=np.float64),
            "loc": np.array([-2.0, -2.0, -2.0], dtype=np.float64),
            "scale": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        }
    )

    # Input 6: 3D tensors
    list_of_inputs.append(
        {
            "x": np.full((2, 3, 4), 0.4, dtype=np.float32),
            "a": np.full((2, 3, 4), 2.5, dtype=np.float32),
            "b": np.full((2, 3, 4), 3.5, dtype=np.float32),
            "loc": np.zeros((2, 3, 4), dtype=np.float32),
            "scale": np.ones((2, 3, 4), dtype=np.float32),
        }
    )

    # Input 7: Small shape parameters (a, b < 1)
    list_of_inputs.append(
        {
            "x": np.array([0.05, 0.5, 0.95], dtype=np.float64),
            "a": np.array([0.2, 0.1, 0.05], dtype=np.float64),
            "b": np.array([0.2, 0.1, 0.05], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )

    # Input 8: Boundary and outside-range values
    list_of_inputs.append(
        {
            "x": np.array([-0.5, 0.0, 1.0, 1.5], dtype=np.float32),
            "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
    )

    # Input 9: Large shape parameters
    list_of_inputs.append(
        {
            "x": np.array([0.48, 0.50, 0.52], dtype=np.float64),
            "a": np.array([100.0, 200.0, 500.0], dtype=np.float64),
            "b": np.array([100.0, 200.0, 500.0], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )

    # Input 10: Broadcasting with scalar-shaped dimensions
    list_of_inputs.append(
        {
            "x": np.linspace(0.1, 0.9, 10, dtype=np.float32).reshape(10, 1),
            "a": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "b": np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
            "loc": np.array([[0.0]], dtype=np.float32),
            "scale": np.array([[1.0]], dtype=np.float32),
        }
    )

    # Input 11: 4D tensor with varying scale
    list_of_inputs.append(
        {
            "x": np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(
                np.float32
            ),
            "a": np.full((2, 2, 2, 2), 3.0, dtype=np.float32),
            "b": np.full((2, 2, 2, 2), 4.0, dtype=np.float32),
            "loc": np.full((2, 2, 2, 2), -1.0, dtype=np.float32),
            "scale": np.full((2, 2, 2, 2), 5.0, dtype=np.float32),
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.logsf"] = logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.logsf'.")


check_valid('jax.scipy.stats.beta.logsf', generated_inputs['jax.scipy.stats.beta.logsf'], lib="jax", suffix=0)
