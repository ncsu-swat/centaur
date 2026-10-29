
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_l_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard parameters
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "x": np.array([-2.5, -0.5, 0.0, 1.2, 3.4], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float64),
        "scale": np.array([0.5, 1.0, 1.5, 2.0, 0.8], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting scalar loc and scale over 1D x
    input_dict = {
        "x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        "loc": np.array(2.0, dtype=np.float32),
        "scale": np.array(3.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays with identical shapes
    input_dict = {
        "x": np.array([[-1.0, 0.5], [2.0, -3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32),
        "scale": np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D with broadcasting along rows
    input_dict = {
        "x": np.zeros((3, 4), dtype=np.float64),
        "loc": np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64),
        "scale": np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D with broadcasting along columns
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "loc": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        "scale": np.array([[1.0], [2.0], [0.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays
    input_dict = {
        "x": np.random.randn(2, 3, 4).astype(np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 2.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer tensors castable to float
    input_dict = {
        "x": np.array([-3, -1, 0, 2, 5], dtype=np.int32),
        "loc": np.array([0, 0, 1, 1, -2], dtype=np.int32),
        "scale": np.array([1, 2, 1, 3, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative and positive extreme values
    input_dict = {
        "x": np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensors
    input_dict = {
        "x": np.ones((2, 2, 2, 2), dtype=np.float32) * 1.5,
        "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_l.logsf"] = gumbel_l_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logsf'.")


check_valid('jax.scipy.stats.gumbel_l.logsf', generated_inputs['jax.scipy.stats.gumbel_l.logsf'], lib="jax", suffix=0)
