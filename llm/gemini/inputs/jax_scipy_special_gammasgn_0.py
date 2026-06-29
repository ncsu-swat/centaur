
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gammasgn_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 1D array of negative floats (non-integers)
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 1D array with special values (+0, -0, inf, -inf, nan)
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 1D array containing negative integers (should return nan)
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 5: 2D array of mixed floats
    x = np.array([[1.2, -0.8], [-2.3, 3.1]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: 3D array
    x = np.array([[[0.5, -1.5], [2.5, -3.5]], [[4.5, -5.5], [6.5, -7.5]]], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 7: 0D array (scalar array)
    x = np.array(-1.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 8: Float64 1D array
    x = np.array([10.5, -10.5, 20.1, -20.1], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 9: Large values
    x = np.array([100.0, -100.5, 500.2, -500.8], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 10: Array with very small non-zero values
    x = np.array([1e-10, -1e-10, 1e-20, -1e-20], dtype=np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.gammasgn"] = gammasgn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammasgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammasgn'.")


check_valid('jax.scipy.special.gammasgn', generated_inputs['jax.scipy.special.gammasgn'], lib="jax", suffix=0)
