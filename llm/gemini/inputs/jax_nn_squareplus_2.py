
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_squareplus_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard parameters
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    b = 4.0
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 2: 2D float32 array, random values
    x = np.random.randn(3, 5).astype(np.float32)
    b = 2.5
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 3: 3D float64 array, small b
    x = np.random.randn(2, 3, 4).astype(np.float64)
    b = 0.1
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 4: 1D float16 array
    x = np.random.randn(10).astype(np.float16)
    b = 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 5: 4D float32 array, large b
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = 100.0
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 6: 0D (scalar) float32 array
    x = np.array(1.5, dtype=np.float32)
    b = 3.14
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 7: 1D linspace array with positive, negative, and zero values, tiny b
    x = np.linspace(-10.0, 10.0, 100, dtype=np.float32)
    b = 1e-4
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 8: 2D float32 array with large magnitude values
    x = np.random.uniform(-1000.0, 1000.0, size=(5, 5)).astype(np.float32)
    b = 50.0
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 9: 2D zero array
    x = np.zeros((3, 3), dtype=np.float32)
    b = 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    # Input 10: 3D float64 array, high dimensions
    x = np.ones((1, 10, 1), dtype=np.float64)
    b = 0.5
    list_of_inputs.append({"x": copy.deepcopy(x), "b": b})

    return list_of_inputs

generated_inputs["jax.nn.squareplus_2"] = jax_nn_squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_2'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_2'], lib="jax", suffix=2)
