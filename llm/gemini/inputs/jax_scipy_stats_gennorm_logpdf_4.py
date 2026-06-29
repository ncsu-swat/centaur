
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "x": 0.0,
        "beta": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "x": 1.5,
        "beta": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "x": -2.3,
        "beta": np.array([1.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "x": 0.5,
        "beta": np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "x": -1.0,
        "beta": np.ones((3, 3), dtype=np.float32) * 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "x": 2.0,
        "beta": np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "x": 0.0,
        "beta": np.array([0.01, 100.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "x": -0.5,
        "beta": np.array([[[1.2, 2.3], [3.4, 4.5]], [[5.6, 6.7], [7.8, 8.9]]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "x": 3.14,
        "beta": np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "x": -4.5,
        "beta": np.linspace(0.5, 4.5, 10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.logpdf_4"] = gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_4'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_4'], lib="jax", suffix=4)
