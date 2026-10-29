
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, odd length, axis 0
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 2: 1D float64 array, even length, axis 0
    x = np.arange(8, dtype=np.float64)
    axes = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 3: 1D complex64 array, negative axis index
    x = (np.arange(6) + 1j * np.arange(6, 12)).astype(np.complex64)
    axes = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 4: 2D float32 array, shift along axis 0
    x = np.random.randn(4, 5).astype(np.float32)
    axes = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 5: 2D float32 array, shift along axis 1
    x = np.random.randn(6, 7).astype(np.float32)
    axes = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 6: 2D complex128 array, negative axis index -2
    x = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex128
    )
    axes = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 7: 3D int32 array, shift along axis 1
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    axes = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 8: 3D float64 array with negative values, shift along axis 2
    x = np.linspace(-10.0, 10.0, 30, dtype=np.float64).reshape(2, 3, 5)
    axes = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 9: 4D float32 array, shift along axis -3
    x = np.random.uniform(-1.0, 1.0, size=(2, 4, 3, 5)).astype(np.float32)
    axes = -3
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 10: 4D int64 array, shift along last axis
    x = np.random.randint(-100, 100, size=(3, 2, 2, 4), dtype=np.int64)
    axes = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 11: 1D bool array
    x = np.array([True, False, True, True, False, False], dtype=bool)
    axes = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftshift_1"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftshift_1'.")


check_valid('jax.numpy.fft.fftshift', generated_inputs['jax.numpy.fft.fftshift_1'], lib="jax", suffix=1)
