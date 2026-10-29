
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atan_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive and negative floats
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array float16
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 2)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 2.0j, -0.5 - 0.5j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D complex128 array
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array float32 with zeros and extreme values
    x = np.array([[[[np.inf, -np.inf], [0.0, -0.0]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array (0 elements) float32
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large values float64
    x = np.array([-1e10, -1e5, 1e5, 1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 3D array with small fractional values
    x = np.linspace(-0.5, 0.5, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array float32 containing NaN
    x = np.array([np.nan, 0.5, -np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.atan"] = atan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atan'.")


check_valid('jax.lax.atan', generated_inputs['jax.lax.atan'], lib="jax", suffix=0)
