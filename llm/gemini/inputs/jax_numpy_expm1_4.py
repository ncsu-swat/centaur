
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    list_of_inputs.append({"x": np.array([1.0, 2.5, 3.1], dtype=np.float32)})

    # Input 2: 1D array of negative floats
    list_of_inputs.append({"x": np.array([-0.1, -1.0, -10.0], dtype=np.float32)})

    # Input 3: 1D array of very small values
    list_of_inputs.append({"x": np.array([1e-5, 1e-10, -1e-12], dtype=np.float32)})

    # Input 4: 2D array of floats
    list_of_inputs.append({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    # Input 5: 2D array with zeros and negatives
    list_of_inputs.append({"x": np.array([[0.0, -0.5], [1.5, -2.0]], dtype=np.float32)})

    # Input 6: 3D array of floats
    list_of_inputs.append({"x": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)})

    # Input 7: 1D array of integers
    list_of_inputs.append({"x": np.array([1, 2, 3, 4], dtype=np.int32)})

    # Input 8: 1D array with special float values
    list_of_inputs.append({"x": np.array([0.0, np.inf, -np.inf, np.nan], dtype=np.float32)})

    # Input 9: 1D array of size 1 (replaces 0-D array to ensure iterability)
    list_of_inputs.append({"x": np.array([0.5], dtype=np.float32)})

    # Input 10: 2D array with shape (1, 3)
    list_of_inputs.append({"x": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)})

    return list_of_inputs

generated_inputs["jax.numpy.expm1_4"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_4'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_4'], lib="jax", suffix=4)
