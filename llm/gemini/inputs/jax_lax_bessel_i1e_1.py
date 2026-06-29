
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with positive values
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: 1D array of float32 with negative values
    x = np.array([-0.5, -1.5, -3.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 2D array of float32 with mixed values
    x = np.array([[1.0, -1.0], [2.5, -2.5]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: 3D array of float64
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 5: 0D array (scalar) of float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 6: Large 1D array of float32
    x = np.linspace(-100.0, 100.0, 50, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 7: 1D array of float64 with very small values
    x = np.array([1e-5, -1e-5, 1e-10, -1e-10], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 8: 4D array of float32
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: 1D array of float32 containing zeros
    x = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 10: 2D array of float64 with large values to test exponential scaling stability
    x = np.array([[500.0, -500.0], [1000.0, -1000.0]], dtype=np.float64)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_1"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_1'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_1'], lib="jax", suffix=1)
