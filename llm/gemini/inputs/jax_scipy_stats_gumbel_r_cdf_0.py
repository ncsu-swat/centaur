
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard float32
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of matching shape, float64
    input_dict = {
        "x": np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float64),
        "loc": np.array([0.0, -1.0, 0.5, 2.0, 1.0], dtype=np.float64),
        "scale": np.array([0.5, 1.0, 2.0, 1.5, 0.8], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with broadcasting (x is (3, 4), loc is (3, 1), scale is (1, 4))
    input_dict = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "loc": np.array([[-1.0], [0.0], [1.0]], dtype=np.float32),
        "scale": np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, all same shape
    input_dict = {
        "x": np.random.uniform(-5.0, 5.0, size=(2, 3, 2)).astype(np.float32),
        "loc": np.zeros((2, 3, 2), dtype=np.float32),
        "scale": np.ones((2, 3, 2), dtype=np.float32) * 2.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar loc and scale broadcasting over 1D x
    input_dict = {
        "x": np.linspace(-10.0, 10.0, 20, dtype=np.float64),
        "loc": np.array(1.5, dtype=np.float64),
        "scale": np.array(3.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values for loc, small scale
    input_dict = {
        "x": np.array([-10.0, -5.0, 0.0], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([0.1, 0.2, 0.05], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive scale values
    input_dict = {
        "x": np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64),
        "loc": np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64),
        "scale": np.array([[50.0, 100.0], [150.0, 200.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor inputs
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(
            np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element 1D arrays
    input_dict = {
        "x": np.array([2.718], dtype=np.float32),
        "loc": np.array([3.1415], dtype=np.float32),
        "scale": np.array([1.414], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting 2D x with 1D scale and 0D loc
    input_dict = {
        "x": np.random.randn(5, 3).astype(np.float64),
        "loc": np.array(0.0, dtype=np.float64),
        "scale": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Extreme negative x values evaluating close to 0 cdf
    input_dict = {
        "x": np.array([-100.0, -50.0, -20.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_r.cdf"] = gumbel_r_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.cdf'.")


check_valid('jax.scipy.stats.gumbel_r.cdf', generated_inputs['jax.scipy.stats.gumbel_r.cdf'], lib="jax", suffix=0)
