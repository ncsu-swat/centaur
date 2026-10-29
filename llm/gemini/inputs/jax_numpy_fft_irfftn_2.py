
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array, backward normalization
    a = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    s = (8,)
    axes = (0,)
    norm = "backward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 2: 2D array, all axes transformed, ortho normalization
    a = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(np.complex64)
    s = (4, 8)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 3: 3D array, complex128 dtype, forward normalization
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex128)
    s = (3, 4, 8)
    axes = (0, 1, 2)
    norm = "forward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 4: 3D array, transformed along subset of negative axes
    a = (np.random.randn(6, 5, 4) + 1j * np.random.randn(6, 5, 4)).astype(np.complex64)
    s = (5, 6)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 5: 4D array, all axes transformed
    a = (np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)).astype(np.complex64)
    s = (2, 3, 4, 8)
    axes = (0, 1, 2, 3)
    norm = "ortho"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 6: 2D array with permuted axes order
    a = (np.random.randn(5, 4) + 1j * np.random.randn(5, 4)).astype(np.complex64)
    s = (6, 4)
    axes = (1, 0)
    norm = "forward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 7: 3D array transformed along a single intermediate axis
    a = (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(np.complex64)
    s = (6,)
    axes = (1,)
    norm = "backward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 8: 2D array with odd-sized output dimension along the real FFT axis
    a = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(np.complex64)
    s = (3, 9)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 9: 4D array with real float32 input
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    s = (4, 4)
    axes = (2, 3)
    norm = "backward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    # Input 10: 3D array with negative values and float64 dtype
    a = -np.abs(np.random.randn(4, 4, 4)).astype(np.float64)
    s = (4, 4, 6)
    axes = (-3, -2, -1)
    norm = "forward"
    list_of_inputs.append(copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm}))

    return list_of_inputs

generated_inputs["jax.numpy.fft.irfftn_2"] = irfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfftn_2'.")


check_valid('jax.numpy.fft.irfftn', generated_inputs['jax.numpy.fft.irfftn_2'], lib="jax", suffix=2)
