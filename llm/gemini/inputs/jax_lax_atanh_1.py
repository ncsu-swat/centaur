
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalar
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 1D float32 array with negative and positive values
    x = np.array([-0.8, -0.3, 0.0, 0.3, 0.8], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 2D float64 array
    x = np.random.uniform(-0.95, 0.95, (3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 4: 3D float32 array
    x = np.random.uniform(-0.99, 0.99, (2, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 5: 4D float64 array
    x = np.random.uniform(-0.5, 0.5, (2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 6: 1D complex64 array (complex values are valid for atanh)
    x = (np.random.uniform(-0.8, 0.8, (5,)) + 1j * np.random.uniform(-0.8, 0.8, (5,))).astype(np.complex64)
    list_of_inputs.append({"x": x})

    # Input 7: 2D complex128 array
    x = (np.random.uniform(-2.0, 2.0, (2, 3)) + 1j * np.random.uniform(-2.0, 2.0, (2, 3))).astype(np.complex128)
    list_of_inputs.append({"x": x})

    # Input 8: 1D float32 array, all negative
    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: 2D float32 array, all positive
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 10: 1D float64 array, close to limits
    x = np.array([-0.999, -0.99, 0.0, 0.99, 0.999], dtype=np.float64)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.lax.atanh_1"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atanh_1'.")


check_valid('jax.lax.atanh', generated_inputs['jax.lax.atanh_1'], lib="jax", suffix=1)
