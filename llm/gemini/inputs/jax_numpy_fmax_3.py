
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "x1": 5.0,
        "x2": np.array([1.0, 7.0, 9.0, 4.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "x1": -2.5,
        "x2": np.array([-5.0, 0.0, 2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "x1": 0.0,
        "x2": np.random.randn(3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "x1": float('nan'),
        "x2": np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "x1": float('inf'),
        "x2": np.random.randn(2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "x1": -100.0,
        "x2": np.linspace(-200.0, 200.0, 10).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "x1": 1.5,
        "x2": np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, -1.0], [5.0, 1.5]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "x1": float('-inf'),
        "x2": np.array([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "x1": 10.0,
        "x2": np.zeros((1, 5), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "x1": -0.5,
        "x2": np.ones((4, 4, 1), dtype=np.float32) * -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmax_3"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_3'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_3'], lib="jax", suffix=3)
