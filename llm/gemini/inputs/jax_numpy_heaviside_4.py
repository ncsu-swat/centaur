
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "x1": 0.0,
        "x2": np.array([0.5, 1.0, 1.5], dtype=np.float32)
    })

    # Input 2
    list_of_inputs.append({
        "x1": -1.5,
        "x2": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    })

    # Input 3
    list_of_inputs.append({
        "x1": 2.5,
        "x2": np.random.randn(3, 3, 3).astype(np.float64)
    })

    # Input 4
    list_of_inputs.append({
        "x1": 0.0,
        "x2": np.array([10, 20], dtype=np.int32)
    })

    # Input 5
    list_of_inputs.append({
        "x1": -0.0,
        "x2": np.ones((5, 5), dtype=np.float32) * 0.25
    })

    # Input 6
    list_of_inputs.append({
        "x1": 1.0,
        "x2": np.array([0, 1, 2], dtype=np.int64)
    })

    # Input 7
    list_of_inputs.append({
        "x1": -100.0,
        "x2": np.random.uniform(-1, 1, size=(2, 4)).astype(np.float32)
    })

    # Input 8
    list_of_inputs.append({
        "x1": 0.5,
        "x2": np.array([[[1.0]]], dtype=np.float32)
    })

    # Input 9
    list_of_inputs.append({
        "x1": -9.9,
        "x2": np.arange(10, dtype=np.float32)
    })

    # Input 10
    list_of_inputs.append({
        "x1": 0.0,
        "x2": np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    })

    # Input 11
    list_of_inputs.append({
        "x1": 4.2,
        "x2": np.ones((2, 2, 2, 2), dtype=np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_4"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_4'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_4'], lib="jax", suffix=4)
