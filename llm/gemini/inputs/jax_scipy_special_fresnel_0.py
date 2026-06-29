
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fresnel_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array in [-10, 10]
    x = np.linspace(-10.0, 10.0, 50, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 2D float64 array in [-5, 5]
    x = np.random.uniform(-5.0, 5.0, (10, 10)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 3: 3D float32 array with negative and positive values
    x = np.random.uniform(-8.0, 8.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 0D array (scalar) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 5: 1D float16 array
    x = np.linspace(-2.0, 2.0, 20, dtype=np.float16)
    list_of_inputs.append({"x": x})

    # Input 6: 4D float64 array
    x = np.random.uniform(-3.0, 3.0, (2, 2, 3, 3)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 7: Array with specific edge cases: negative, zero, positive
    x = np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 8: Array with very small values near zero
    x = np.random.uniform(-1e-4, 1e-4, (5, 5)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: Array with large values (outside guaranteed domain, but still valid types)
    x = np.linspace(-20.0, 20.0, 10, dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 10: 1D array of uniform random values in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, 100).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 11: 5D array
    x = np.random.uniform(-2.0, 2.0, (2, 2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.fresnel"] = fresnel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.fresnel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.fresnel'.")


check_valid('jax.scipy.special.fresnel', generated_inputs['jax.scipy.special.fresnel'], lib="jax", suffix=0)
