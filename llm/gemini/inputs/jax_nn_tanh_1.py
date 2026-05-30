
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array (matrix)
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D array) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int32 array (should promote to float in JAX)
    x = np.arange(-10, 10).reshape(2, 2, 5).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array to test complex hyperbolic tangent
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float16 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex128 array
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 5D float32 array (higher dimensions)
    x = np.random.randn(2, 1, 3, 1, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large values to test numerical saturation bounds
    x = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array with special values (NaN and Infinities)
    x = np.array([np.nan, -np.inf, np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array with specified shape
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.tanh_1"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.tanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.tanh_1'.")


check_valid('jax.nn.tanh', generated_inputs['jax.nn.tanh_1'], lib="jax", suffix=1)
