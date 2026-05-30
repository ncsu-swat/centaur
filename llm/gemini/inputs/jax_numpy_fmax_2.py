
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive float
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 2.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: 2D float64 array with negative values, negative float
    x1 = np.array([[-1.0, -2.5], [-3.0, -4.2]], dtype=np.float64)
    x2 = -2.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: 3D int32 array, float
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = 4.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: 1D array containing NaNs and Infs, positive float
    x1 = np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32)
    x2 = 5.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: 2D array, nan float
    x1 = np.array([[-np.inf, np.nan], [0.0, 5.0]], dtype=np.float32)
    x2 = float('nan')
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: 3D random float32 array, zero float
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = 0.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: 0D scalar-like array, float
    x1 = np.array(5.0, dtype=np.float32)
    x2 = 3.14
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: 1D linspace float64 array, negative float
    x1 = np.linspace(-10, 10, 50).astype(np.float64)
    x2 = -1.5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: 2D float64 array, large float
    x1 = np.array([[10.5, -20.3, 0.0], [100.2, -50.1, 10.0]], dtype=np.float64)
    x2 = 1000.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: 4D random float32 array, float boundary value
    x1 = np.random.uniform(-100, 100, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = 50.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmax_2"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_2'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_2'], lib="jax", suffix=2)
