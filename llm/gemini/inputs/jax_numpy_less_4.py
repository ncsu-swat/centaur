
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def less_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    x = 0.5
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D float32 array with random normal values
    x = -1.2
    y = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: 3D float32 array with random values
    x = 10.0
    y = np.random.uniform(-20, 20, size=(3, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: 1D float64 array using linspace
    x = 3.14159
    y = np.linspace(0, 5, 10).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: 2D float32 array with uniform values, x is negative zero
    x = -0.0
    y = np.random.uniform(-1, 1, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: 4D float32 array
    x = 100.0
    y = (np.ones((2, 2, 2, 2), dtype=np.float32) * 50.0)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: 1D float32 array from arange
    x = -50.5
    y = np.arange(-100, 100, 10, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: 2D float64 array with exponential distribution
    x = 1e-3
    y = np.random.exponential(scale=1.0, size=(4, 4)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: 2D array with 1 row and 5 columns
    x = 2.718
    y = np.random.normal(0, 1, size=(1, 5)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: 2D float32 array with zeros
    x = 1.5
    y = np.zeros((3, 1), dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.less_4"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_4'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_4'], lib="jax", suffix=4)
