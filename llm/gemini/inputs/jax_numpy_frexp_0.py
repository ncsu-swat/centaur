
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def frexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive float32 values
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of negative float32 values
    x = np.array([-1.5, -2.5, -3.75, -8.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float32)
    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of float64 values with positive, negative, and zero
    x = np.array([[-0.0, 0.0, 0.5], [1024.0, -1024.0, 2.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of float32 values
    x = np.random.uniform(0.1, 100.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array of float16 values
    x = np.array([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array of float32 values
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with very small and large float64 numbers
    x = np.array([1e-30, 1e-10, 1.0, 1e10, 1e30], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array of powers of two (float32)
    x = np.array([[0.25, 0.5], [1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with subnormal/tiny float32 values and zeros
    x = np.array([0.0, -0.0, 1e-38, -1e-38], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D array of float64 values with mixed signs
    x = np.random.randn(3, 1, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.frexp"] = frexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.frexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.frexp'.")


check_valid('jax.numpy.frexp', generated_inputs['jax.numpy.frexp'], lib="jax", suffix=0)
