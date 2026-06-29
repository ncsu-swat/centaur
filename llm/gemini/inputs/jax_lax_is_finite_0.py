
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def is_finite_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array with finite values
    x = np.array([-1.2, 0.0, 3.4, 5.6], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float32 array containing inf, -inf, and nan
    x = np.array([1.0, np.inf, -np.inf, np.nan, 2.3], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float64 array with random normal values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array containing nan and inf values
    x = np.array([[[1.0, np.nan], [np.inf, 4.0]], [[-np.inf, 6.0], [7.0, 8.0]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D float16 array with small numbers
    x = np.array([1e-3, -2e-4, 0.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D float32 array (scalar tensor) with finite value
    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 0D float32 array (scalar tensor) with nan
    x = np.array(np.nan, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float64 array with random uniform values
    x = np.random.uniform(-10.0, 10.0, (2, 2, 3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D float32 array with large finite values and infinities
    x = np.array([[3.4028235e+38, -3.4028235e+38], [np.inf, -np.inf]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float16 array of zeros
    x = np.zeros((2, 1, 2, 1, 3), dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.is_finite"] = is_finite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.is_finite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.is_finite'.")


check_valid('jax.lax.is_finite', generated_inputs['jax.lax.is_finite'], lib="jax", suffix=0)
