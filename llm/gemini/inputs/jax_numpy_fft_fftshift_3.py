
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, single axis
    x = np.linspace(-10.0, 10.0, 10, dtype=np.float32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 2: 1D int32 array with negative values, negative axis
    x = np.arange(-5, 6, dtype=np.int32)
    axes = (-1,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 3: 2D float64 array, both axes
    x = np.random.randn(6, 8).astype(np.float64)
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 4: 2D complex64 array, single axis
    x = (
        np.random.randn(5, 9).astype(np.float32)
        + 1j * np.random.randn(5, 9).astype(np.float32)
    ).astype(np.complex64)
    axes = (1,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 5: 2D complex128 array, negative axis
    x = (
        np.random.randn(4, 4).astype(np.float64)
        + 1j * np.random.randn(4, 4).astype(np.float64)
    ).astype(np.complex128)
    axes = (-2,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 6: 3D float32 array, non-adjacent axes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axes = (0, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 7: 3D float64 array, all negative axes
    x = np.random.randn(5, 5, 5).astype(np.float64)
    axes = (-3, -2, -1)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 8: 4D complex64 array, subset of axes
    x = (
        np.random.randn(2, 4, 3, 5).astype(np.float32)
        + 1j * np.random.randn(2, 4, 3, 5).astype(np.float32)
    ).astype(np.complex64)
    axes = (1, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 9: 4D int64 array, mixed sign axes
    x = np.random.randint(-100, 100, size=(3, 2, 2, 4), dtype=np.int64)
    axes = (-4, -1)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 10: 5D float32 array, all axes
    x = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    axes = (0, 1, 2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 11: 2D float32 array, empty axes tuple (no shift)
    x = np.random.randn(3, 3).astype(np.float32)
    axes = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftshift_3"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftshift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftshift_3'.")


check_valid('jax.numpy.fft.fftshift', generated_inputs['jax.numpy.fft.fftshift_3'], lib="jax", suffix=3)
