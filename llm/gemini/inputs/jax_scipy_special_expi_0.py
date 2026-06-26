
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expi_inputs():
    list_of_inputs = []

    # Input 1: 1D array, small positive float32 values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 1D array, negative float32 values
    x = np.array([-1.0, -2.5, -0.5], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 2D array, mixed positive and negative values
    x = np.array([[-1.5, 0.5], [2.0, -3.0]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 1D array, float64 values
    x = np.array([0.1, 1.5, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 5: 3D array, random float32 values
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: Scalar represented as 0D array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 7: 1D array, very small values close to zero (excluding zero)
    x = np.array([-1e-4, 1e-4], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 8: 1D array, large positive values
    x = np.array([5.0, 15.0, 30.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: 1D array, large negative values
    x = np.array([-10.0, -30.0, -50.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 10: 4D array, random float64 values
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.expi"] = expi_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.expi' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.expi'.")


check_valid('jax.scipy.special.expi', generated_inputs['jax.scipy.special.expi'], lib="jax", suffix=0)
