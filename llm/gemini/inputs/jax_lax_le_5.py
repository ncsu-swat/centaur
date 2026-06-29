
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, float y
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = 0.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, float y
    x = np.random.randn(3, 3).astype(np.float32)
    y = 1.5
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, float y
    x = np.random.randn(2, 2, 2).astype(np.float64)
    y = -0.5
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array, float y
    x = np.random.randn(2, 3, 4, 1).astype(np.float32)
    y = 10.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D float32 array, float y
    x = np.array(5.5, dtype=np.float32)
    y = 5.5
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array with specific range, float y
    x = np.linspace(-10, 10, 12).reshape(3, 4).astype(np.float64)
    y = -2.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with positive and negative infinity, float y
    x = np.array([-np.inf, np.inf, 0.0], dtype=np.float32)
    y = 1.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array, float y
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    y = 0.1
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, float y
    x = np.random.uniform(-100, 100, (5, 5)).astype(np.float64)
    y = 50.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, float y
    x = np.ones((2, 3, 2), dtype=np.float32) * -5.0
    y = -5.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_5"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_5'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_5'], lib="jax", suffix=5)
