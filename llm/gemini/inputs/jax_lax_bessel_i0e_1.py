
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive values (float32)
    x = np.array([0.1, 0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative values (float64)
    x = np.array([-10.0, -5.0, -2.5, -1.0, -0.1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array with mixed values (float32)
    x = np.array([[-3.0, -1.5, 0.0], [1.5, 3.0, 4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with large values (float64)
    x = np.array(
        [[50.0, 100.0], [200.0, 500.0], [-100.0, -500.0]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array with small values near zero (float32)
    x = np.array(
        [[[1e-4, -1e-4], [2e-5, -2e-5]], [[3e-6, -3e-6], [0.0, 1e-7]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array with shape (2, 1, 4) (float64)
    x = np.linspace(-10.0, 10.0, 8, dtype=np.float64).reshape(2, 1, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array from normal distribution (float32)
    x = np.random.randn(2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with integer-valued floats (float64)
    x = np.arange(-5, 6, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D uniform random array (float32)
    x = np.random.uniform(-20.0, 20.0, size=(4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.bessel_i0e_1"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_1'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_1'], lib="jax", suffix=1)
