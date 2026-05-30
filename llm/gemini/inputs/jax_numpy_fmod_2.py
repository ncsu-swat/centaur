
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array, positive float divisor
    x1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D float32 array with negative values, positive float divisor
    x1 = np.array([-1.5, -2.5, 3.5, -4.5], dtype=np.float32)
    x2 = 1.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 2D float64 array, negative float divisor
    x1 = np.array([[5.0, -6.0], [7.0, -8.0]], dtype=np.float64)
    x2 = -3.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D int32 array, fractional float divisor
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = 3.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 3D float32 array, fractional float divisor
    x1 = np.random.uniform(-10, 10, (2, 2, 2)).astype(np.float32)
    x2 = 2.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Large 1D array, large float divisor
    x1 = np.linspace(-100, 100, 10).astype(np.float32)
    x2 = 12.34
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 4D float32 array, small float divisor
    x1 = np.random.randn(2, 3, 2, 1).astype(np.float32)
    x2 = 0.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: int16 array with negative values, float divisor
    x1 = np.array([[-10, 15], [-20, 25]], dtype=np.int16)
    x2 = 4.2
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: float64 1D array, negative small divisor
    x1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    x2 = -0.15
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 5D array, integer-like float divisor
    x1 = np.ones((1, 2, 1, 2, 1), dtype=np.float32) * 5.5
    x2 = 2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmod_2"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_2'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_2'], lib="jax", suffix=2)
