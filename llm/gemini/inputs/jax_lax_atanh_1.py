
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atanh_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive and negative values (float32)
    x = np.array([-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array (float64)
    x = np.array([[-0.75, -0.25], [0.25, 0.75]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array with small random values in (-1, 1) (float32)
    x = np.random.uniform(-0.9, 0.9, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array (float16)
    x = np.random.uniform(-0.5, 0.5, size=(2, 2, 2, 2)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array of complex numbers (complex64)
    x = np.array(
        [0.5 + 0.5j, -0.5 + 0.2j, 0.0 - 0.7j, 1.2 + 0.3j], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D array of complex numbers (complex128)
    x = np.array(
        [[0.2 - 0.1j, -0.4 + 0.6j], [0.8 + 0.0j, -0.1 - 0.9j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D empty array (float32)
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with single row (float64)
    x = np.array([[-0.99, -0.5, 0.0, 0.5, 0.99]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 3D array with zeros (float32)
    x = np.zeros((3, 1, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with values near zero (float64)
    x = np.array([-1e-7, -1e-15, 0.0, 1e-15, 1e-7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

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
