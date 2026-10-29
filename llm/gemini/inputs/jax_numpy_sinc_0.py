
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def sinc_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive and negative floats, including zero
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: Scalar (0-D array) float
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64 values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array of integers (will be promoted to inexact type)
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Float16 array
    x = np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex64 array
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex128 2D array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with large values (positive and negative)
    x = np.array([-1e5, -100.0, 0.0, 100.0, 1e5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array (0-length dimension)
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sinc"] = sinc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sinc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sinc'.")


check_valid('jax.numpy.sinc', generated_inputs['jax.numpy.sinc'], lib="jax", suffix=0)
