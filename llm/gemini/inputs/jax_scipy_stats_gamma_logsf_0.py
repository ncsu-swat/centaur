
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard float32
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "a": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with matching shapes, float64
    input_dict = {
        "x": np.array([0.5, 1.5, 3.0, 5.0], dtype=np.float64),
        "a": np.array([1.0, 2.0, 2.5, 4.0], dtype=np.float64),
        "loc": np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float64),
        "scale": np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "a": np.array([[2.0, 1.5], [0.5, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.5], [2.0, 0.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 1D parameters with 2D x
    input_dict = {
        "x": np.array([[0.5, 1.2, 3.4], [2.1, 0.8, 4.5]], dtype=np.float32),
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Values of x less than loc (testing boundary/tail behavior)
    input_dict = {
        "x": np.array([-2.0, -0.5, 0.0, 1.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor inputs
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float64) * 5.0,
        "a": np.ones((2, 3, 4), dtype=np.float64) * 2.5,
        "loc": np.zeros((2, 3, 4), dtype=np.float64),
        "scale": np.ones((2, 3, 4), dtype=np.float64) * 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small shape parameter `a` < 1
    input_dict = {
        "x": np.array([0.01, 0.1, 0.5, 1.0], dtype=np.float32),
        "a": np.array([0.1, 0.2, 0.5, 0.8], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large x values (deep tail evaluation)
    input_dict = {
        "x": np.array([20.0, 50.0, 100.0], dtype=np.float64),
        "a": np.array([5.0, 10.0, 15.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative loc offset
    input_dict = {
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed shapes with scalar-like 1D array broadcasting
    input_dict = {
        "x": np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        "a": np.array([1.5], dtype=np.float32),
        "loc": np.array([0.5], dtype=np.float32),
        "scale": np.array([0.8], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gamma.logsf"] = gamma_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logsf'.")


check_valid('jax.scipy.stats.gamma.logsf', generated_inputs['jax.scipy.stats.gamma.logsf'], lib="jax", suffix=0)
