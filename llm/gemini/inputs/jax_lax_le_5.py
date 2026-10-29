
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lax_le_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive float
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D array with zero threshold
    x = np.array([[-2.5, 3.1], [0.0, -0.1]], dtype=np.float32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D array with negative float
    x = np.ones((2, 3, 4), dtype=np.float32) * -5.0
    y = -2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar array
    x = np.array(4.2, dtype=np.float32)
    y = 4.2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 4D array with large float
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = 100.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D array with negative values and negative float
    x = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    y = -15.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D array containing inf and -inf
    x = np.array([[np.inf, -np.inf], [0.0, np.nan]], dtype=np.float32)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D empty array
    x = np.array([], dtype=np.float32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D row vector
    x = np.linspace(-1.0, 1.0, 5, dtype=np.float32).reshape(1, 5)
    y = -0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 5D array with fractional float
    x = np.zeros((1, 2, 1, 3, 1), dtype=np.float32)
    y = 3.141592653589793
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D column vector with very small float
    x = np.array([[1e-5], [1e-6], [1e-7]], dtype=np.float32)
    y = 1e-6
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.lax.le_5"] = lax_le_inputs()

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
