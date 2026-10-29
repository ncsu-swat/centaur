
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_l_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar tensors, float32
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensors, same shape, float32
    input_dict = {
        "x": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.5, 2.0, 0.5, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensors, same shape, float64
    input_dict = {
        "x": np.array([[-1.5, 0.2], [3.1, -0.7]], dtype=np.float64),
        "loc": np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float64),
        "scale": np.array([[0.8, 1.2], [2.0, 0.4]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with 1D and scalar tensors
    input_dict = {
        "x": np.array([-3.0, -1.0, 0.5, 2.5], dtype=np.float32),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array([2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 2D with 1D
    input_dict = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "loc": np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        "scale": np.array([[1.0], [2.0], [0.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors, float64
    input_dict = {
        "x": np.random.randn(2, 3, 2).astype(np.float64),
        "loc": np.zeros((2, 3, 2), dtype=np.float64),
        "scale": np.ones((2, 3, 2), dtype=np.float64) * 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive and negative x values
    input_dict = {
        "x": np.array([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=np.float32),
        "loc": np.array([-5.0, 0.0, 0.0, 5.0, 10.0], dtype=np.float32),
        "scale": np.array([10.0, 5.0, 1.0, 2.0, 15.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "loc": np.array([0.0, 0.1, 0.2], dtype=np.float64),
        "scale": np.array([1e-3, 1e-2, 1e-1], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensors
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float32) + 0.1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed negative loc and positive scale
    input_dict = {
        "x": np.array([[-5.0, -2.5], [1.0, 4.5]], dtype=np.float32),
        "loc": np.array([[-10.0, -5.0], [-2.0, 0.0]], dtype=np.float32),
        "scale": np.array([[3.0, 2.0], [1.0, 0.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D with trailing singleton dimensions
    input_dict = {
        "x": np.arange(-3.0, 3.0, 1.0, dtype=np.float64).reshape(6, 1),
        "loc": np.array([[0.0]], dtype=np.float64),
        "scale": np.array([[2.5]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
