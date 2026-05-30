
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def mish_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, basic mix of negative, zero, positive
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array, float32, random values
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array, float32, random values
    x = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar tensor), float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array, float64
    x = np.random.randn(10).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array, float16
    x = np.random.randn(2, 2, 4, 4).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Larger 2D array, float32
    x = np.random.randn(128, 128).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array of zeros, float32
    x = np.zeros((5, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with large magnitudes (both positive and negative)
    x = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array, float32
    x = np.random.randn(2, 1, 3, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.mish"] = mish_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.mish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.mish'.")


check_valid('jax.nn.mish', generated_inputs['jax.nn.mish'], lib="jax", suffix=0)
