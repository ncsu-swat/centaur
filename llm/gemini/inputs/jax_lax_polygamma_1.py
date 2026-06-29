
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: m=0.0, 1D float32 array
    input_dict = {
        "m": 0.0,
        "x": np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: m=1.0, 2D float32 array
    input_dict = {
        "m": 1.0,
        "x": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: m=2.0, 3D float32 array
    input_dict = {
        "m": 2.0,
        "x": np.ones((2, 2, 2), dtype=np.float32) * 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: m=3.0, 1D float64 array
    input_dict = {
        "m": 3.0,
        "x": np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: m=0.0, 4D float32 array
    input_dict = {
        "m": 0.0,
        "x": np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: m=4.0, 2D float64 array
    input_dict = {
        "m": 4.0,
        "x": np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: m=1.0, 0D (scalar) float32 array
    input_dict = {
        "m": 1.0,
        "x": np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: m=2.0, 1D float32 array with negative non-integer values
    input_dict = {
        "m": 2.0,
        "x": np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: m=5.0, 2D float32 array, large values
    input_dict = {
        "m": 5.0,
        "x": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: m=0.0, 3D float64 array
    input_dict = {
        "m": 0.0,
        "x": np.random.uniform(0.5, 2.5, size=(2, 3, 2)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.polygamma_1"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.polygamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.polygamma_1'.")


check_valid('jax.lax.polygamma', generated_inputs['jax.lax.polygamma_1'], lib="jax", suffix=1)
