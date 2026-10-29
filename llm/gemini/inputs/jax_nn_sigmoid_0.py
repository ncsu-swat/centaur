
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sigmoid_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, and zero values (float32)
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array (float32)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array with float64 precision
    x = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array (float32)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array (float32)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with extreme values (float32)
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D array of float16
    x = np.random.randn(4, 4).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array (float64)
    x = np.linspace(-10.0, 10.0, 20, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D empty array (float32)
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array (float32)
    x = np.random.randn(1, 2, 2, 3, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: 2D complex numbers array (complex64)
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.sigmoid"] = sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sigmoid'.")


check_valid('jax.nn.sigmoid', generated_inputs['jax.nn.sigmoid'], lib="jax", suffix=0)
