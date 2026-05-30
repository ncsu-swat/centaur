
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-1.5, -0.7, 0.0, 0.7, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array([[-2.3, 3.8], [4.1, -0.1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 3D float32 array
    x = np.random.uniform(-10, 10, (2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar (0-dim array) float32
    x = np.array(-5.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float64 array with values near integers
    x = np.array([-2.0001, -1.9999, 0.0, 1.9999, 2.0001], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D float32 array with positive and negative values
    x = np.array([[-12.5, 5.5], [0.1, -0.9]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float32 array
    x = np.random.uniform(-100, 100, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D float32 array containing NaN and Inf
    x = np.array([-np.inf, -1.5, np.nan, 1.5, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D float64 array with extreme values
    x = np.array([[-1e10, 1e10], [-1.2345e5, 9.8765e6]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D float32 array with very small values
    x = np.array([-1e-5, 1e-5, -1e-20, 1e-20], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D float64 array
    x = np.random.uniform(-5, 5, (2, 2, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["jax.numpy.floor_1"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_1'.")


check_valid('jax.numpy.floor', generated_inputs['jax.numpy.floor_1'], lib="jax", suffix=1)
