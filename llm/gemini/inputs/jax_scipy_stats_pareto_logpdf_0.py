
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard values within support
    input_dict = {
        "x": np.array([1.5, 2.0, 3.5], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D scalar arrays (rank 0), float64
    input_dict = {
        "x": np.array(2.5, dtype=np.float64),
        "b": np.array(1.2, dtype=np.float64),
        "loc": np.array(0.5, dtype=np.float64),
        "scale": np.array(1.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with negative loc
    input_dict = {
        "x": np.array([[-0.5, 1.0], [2.0, 3.5]], dtype=np.float32),
        "b": np.array([[1.0, 2.0], [3.0, 0.5]], dtype=np.float32),
        "loc": np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes: 2D x and 1D b, loc, scale
    input_dict = {
        "x": np.ones((3, 4), dtype=np.float32) * 5.0,
        "b": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Values outside support (x < loc + scale)
    input_dict = {
        "x": np.array([0.5, 0.8, 1.2], dtype=np.float32),
        "b": np.array([2.5, 2.5, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors
    input_dict = {
        "x": np.full((2, 2, 2), 4.0, dtype=np.float64),
        "b": np.full((2, 2, 2), 1.5, dtype=np.float64),
        "loc": np.zeros((2, 2, 2), dtype=np.float64),
        "scale": np.full((2, 2, 2), 2.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Varying scale and shapes
    input_dict = {
        "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "b": np.array([3.0, 4.0, 5.0], dtype=np.float32),
        "loc": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "scale": np.array([5.0, 5.0, 5.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array broadcasting with scalars
    input_dict = {
        "x": np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32),
        "b": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large parameter values
    input_dict = {
        "x": np.array([100.0, 200.0], dtype=np.float64),
        "b": np.array([50.0, 100.0], dtype=np.float64),
        "loc": np.array([10.0, 20.0], dtype=np.float64),
        "scale": np.array([10.0, 10.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensors
    input_dict = {
        "x": np.ones((1, 2, 1, 3), dtype=np.float32) * 3.0,
        "b": np.ones((1, 2, 1, 3), dtype=np.float32) * 2.0,
        "loc": np.zeros((1, 2, 1, 3), dtype=np.float32),
        "scale": np.ones((1, 2, 1, 3), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.logpdf"] = pareto_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logpdf'.")


check_valid('jax.scipy.stats.pareto.logpdf', generated_inputs['jax.scipy.stats.pareto.logpdf'], lib="jax", suffix=0)
