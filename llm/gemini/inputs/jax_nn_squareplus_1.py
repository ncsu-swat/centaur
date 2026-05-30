
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squareplus_inputs():
    list_of_inputs = []

    # Input 1: 1D array with mixed values, standard b=4
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    b = 4
    list_of_inputs.append({"x": x, "b": b})

    # Input 2: 2D array, b=1
    x = np.random.randn(3, 3).astype(np.float32)
    b = 1
    list_of_inputs.append({"x": x, "b": b})

    # Input 3: 3D array, b=10
    x = np.random.randn(2, 2, 2).astype(np.float32)
    b = 10
    list_of_inputs.append({"x": x, "b": b})

    # Input 4: float64 array, b=5
    x = np.random.randn(5, 5).astype(np.float64)
    b = 5
    list_of_inputs.append({"x": x, "b": b})

    # Input 5: float16 array, b=2
    x = np.random.randn(4).astype(np.float16)
    b = 2
    list_of_inputs.append({"x": x, "b": b})

    # Input 6: Large 1D array, large integer b
    x = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    b = 50
    list_of_inputs.append({"x": x, "b": b})

    # Input 7: 0D array (scalar), b=3
    x = np.array(1.5, dtype=np.float32)
    b = 3
    list_of_inputs.append({"x": x, "b": b})

    # Input 8: 4D array, b=8
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    b = 8
    list_of_inputs.append({"x": x, "b": b})

    # Input 9: Array with large values (positive and negative), b=20
    x = np.array([-100.0, 100.0, -1000.0, 1000.0], dtype=np.float32)
    b = 20
    list_of_inputs.append({"x": x, "b": b})

    # Input 10: 5D array, b=4
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    b = 4
    list_of_inputs.append({"x": x, "b": b})

    return list_of_inputs

generated_inputs["jax.nn.squareplus_1"] = squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_1'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_1'], lib="jax", suffix=1)
