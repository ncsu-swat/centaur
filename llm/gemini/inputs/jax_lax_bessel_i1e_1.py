
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive values
    x = np.array([0.0, 0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative values
    x = np.array([-5.0, -2.5, -1.0, -0.5, -0.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array with float64 values
    x = np.linspace(-10.0, 10.0, num=15, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float32
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array float64
    x = np.random.uniform(-50.0, 50.0, size=(5, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array with large values
    x = np.array([[[100.0, -100.0], [500.0, -500.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 3D array with random normal values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D array float64
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array near zero
    x = np.array([-1e-7, 0.0, 1e-7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

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
