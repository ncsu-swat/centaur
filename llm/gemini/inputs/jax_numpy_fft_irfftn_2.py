
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfftn_inputs():
    list_of_inputs = []

    # Input 1: 3D complex array, transforming all axes
    a = (np.random.randn(4, 4, 3) + 1j * np.random.randn(4, 4, 3)).astype(np.complex64)
    s = (4, 4, 4)
    axes = (0, 1, 2)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D complex array, transforming all axes, ortho norm
    a = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex64)
    s = (8, 8)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3: 3D complex array, transforming last 2 axes, forward norm
    a = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex64)
    s = (3, 4)
    axes = (1, 2)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4: 3D float array, backward norm
    a = np.random.randn(2, 2, 3).astype(np.float32)
    s = (2, 4)
    axes = (1, 2)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5: 2D complex array, ortho norm, odd last dimension output
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    s = (6, 11)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6: 4D complex128 array
    a = (np.random.randn(4, 4, 4, 3) + 1j * np.random.randn(4, 4, 4, 3)).astype(np.complex128)
    s = (4, 4, 4)
    axes = (1, 2, 3)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7: 2D complex64, backward norm
    a = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex64)
    s = (8, 4)
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8: 3D float64 array, ortho norm
    a = np.random.randn(10, 10, 6).astype(np.float64)
    s = (10, 10)
    axes = (1, 2)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9: 4D complex64, transforming 3 axes
    a = (np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)).astype(np.complex64)
    s = (3, 4, 8)
    axes = (1, 2, 3)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10: 2D float32 array
    a = np.random.randn(8, 2).astype(np.float32)
    s = (8, 2)
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 11: 1D complex64 array (with tuple axes of size 1)
    a = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    s = (8,)
    axes = (0,)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

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
