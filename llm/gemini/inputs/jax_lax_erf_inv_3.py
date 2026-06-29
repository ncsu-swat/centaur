
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erf_inv_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar 0.0
    list_of_inputs.append({"x": np.float32(0.0)})

    # Input 2: float64 scalar 0.5
    list_of_inputs.append({"x": np.float64(0.5)})

    # Input 3: float32 scalar -0.5
    list_of_inputs.append({"x": np.float32(-0.5)})

    # Input 4: 1D array of float32
    list_of_inputs.append({"x": np.array([0.0, 0.1, -0.1], dtype=np.float32)})

    # Input 5: 1D array of float64
    list_of_inputs.append({"x": np.array([-0.9, 0.9, 0.0], dtype=np.float64)})

    # Input 6: 2D array of float32
    list_of_inputs.append({"x": np.array([[0.1, 0.2], [-0.3, -0.4]], dtype=np.float32)})

    # Input 7: 2D array of float64
    list_of_inputs.append({"x": np.array([[-0.5, 0.5], [0.0, -0.1]], dtype=np.float64)})

    # Input 8: 3D array of float32
    list_of_inputs.append({"x": np.array([[[0.1, -0.1], [0.2, -0.2]]], dtype=np.float32)})

    # Input 9: float32 scalar near 1
    list_of_inputs.append({"x": np.float32(0.99)})

    # Input 10: float32 scalar near -1
    list_of_inputs.append({"x": np.float32(-0.99)})

    return list_of_inputs

generated_inputs["jax.lax.erf_inv_3"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_3'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_3'], lib="jax", suffix=3)
