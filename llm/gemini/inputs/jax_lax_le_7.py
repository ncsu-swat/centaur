
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def le_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive, negative, and zero values
    input_dict = {
        "x": 0.0,
        "y": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    input_dict = {
        "x": 1.5,
        "y": np.array([[0.0, 1.5], [2.0, -3.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with negative float
    input_dict = {
        "x": -2.5,
        "y": np.random.randn(3, 4, 5).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with larger numbers
    input_dict = {
        "x": 100.0,
        "y": np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D scalar array
    input_dict = {
        "x": 0.5,
        "y": np.array(0.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array of zeros with negative zero float
    input_dict = {
        "x": -0.0,
        "y": np.zeros((2, 2, 2, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Positive infinity float with inf/-inf array
    input_dict = {
        "x": float("inf"),
        "y": np.array([-np.inf, 0.0, np.inf], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative infinity float
    input_dict = {
        "x": float("-inf"),
        "y": np.array([-np.inf, -1e10, 0.0, np.inf], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with broadcast-like dimension
    input_dict = {
        "x": 3.14159,
        "y": np.ones((5, 1, 3), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D reshaped array with small scientific notation float
    input_dict = {
        "x": -1e-5,
        "y": np.linspace(-1.0, 1.0, 10, dtype=np.float32).reshape(2, 5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.le_7"] = le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_7'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_7'], lib="jax", suffix=7)
