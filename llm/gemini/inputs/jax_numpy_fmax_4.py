
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive and negative
    x1 = np.array([-1.5, 2.3, 0.0, -5.7, 10.2], dtype=np.float32)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: 2D int32 array
    x1 = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    x2 = 0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: 3D float64 array
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = -1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: 1D float32 with NaN and Inf
    x1 = np.array([np.nan, np.inf, -np.inf, 3.5, np.nan], dtype=np.float32)
    x2 = 5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: 4D int32 array
    x1 = np.ones((2, 2, 2, 2), dtype=np.int32) * 5
    x2 = 10
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: 2D float32 with negative values
    x1 = np.array([[-10.5, -20.5], [-30.5, -40.5]], dtype=np.float32)
    x2 = -25
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: 1D float32 array
    x1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    x2 = 25
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: 0D array (scalar array) of int32
    x1 = np.array(42, dtype=np.int32)
    x2 = 50
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: Large 2D float32 array
    x1 = np.random.uniform(-100, 100, size=(5, 5)).astype(np.float32)
    x2 = 0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: 1D int32 array with small values
    x1 = np.array([-128, 0, 127], dtype=np.int32)
    x2 = -5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmax_4"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_4'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_4'], lib="jax", suffix=4)
