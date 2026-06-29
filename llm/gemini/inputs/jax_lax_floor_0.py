
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(3, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array
    x = np.random.uniform(-100.0, 100.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D float16 array
    x = np.array([-10.1, -5.5, 0.1, 5.9, 10.2], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D (scalar) float32 array
    x = np.array(-3.7, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float32 array with larger dimensions
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float64 array with values very close to integers
    x = np.array([-1.00001, -0.99999, 0.00001, 0.99999, 1.00001], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D float32 array representing exact integers
    x = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D float32 array with large values
    x = np.array([-123456.78, -0.0001, 0.0001, 123456.78], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 3D float16 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.floor"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.floor'.")


check_valid('jax.lax.floor', generated_inputs['jax.lax.floor'], lib="jax", suffix=0)
