
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def square_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of float32
    x = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array of float64
    x = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array of int32
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar)
    x = np.array(3.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of float32
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of float64 with zeros
    x = np.array([[0.0, -1.5], [2.5, 0.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array of int64
    x = np.array([-100, 50, 0, 200], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D array of int32
    x = np.array([[-3, -2], [2, 3]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array of float32
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.square"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.square'.")


check_valid('jax.lax.square', generated_inputs['jax.lax.square'], lib="jax", suffix=0)
