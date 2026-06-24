
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def i0e_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive float32
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, negative float32
    x = np.array([-0.5, -1.5, -3.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, mixed float64
    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar), float32 positive
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar), float64 negative
    x = np.array(-2.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D array, random float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array, zeroes float32
    x = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array, small values float32
    x = (np.random.rand(2, 2, 2, 2) * 1e-3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D array, large values (testing exponential scaling behavior) float32
    x = np.array([[-100.0, -50.0], [50.0, 100.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array, single element float64
    x = np.array([0.1], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 3D array, large float64
    x = (np.random.rand(3, 3, 3) * 1000 - 500).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i0e'.")


check_valid('jax.scipy.special.i0e', generated_inputs['jax.scipy.special.i0e'], lib="jax", suffix=0)
