
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def silu_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array (matrix)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.randn(2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 4D float32 array (often used in conv nets)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D float64 array for higher precision
    x = np.random.randn(10).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float16 array for lower precision/mixed precision
    x = np.random.randn(5).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 0D array (scalar)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Large 1D array
    x = np.linspace(-10.0, 10.0, 1000, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D array with large extreme values
    x = np.array([[-100.0, -50.0], [50.0, 100.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 3D array of all zeros
    x = np.zeros((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 2D array with very small values
    x = np.random.uniform(-0.01, 0.01, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.silu"] = silu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.silu'.")


check_valid('jax.nn.silu', generated_inputs['jax.nn.silu'], lib="jax", suffix=0)
