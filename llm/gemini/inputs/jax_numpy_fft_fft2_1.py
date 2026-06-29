
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default axes, backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    s = (8, 8)
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D float64 array, cropped size, ortho norm
    a = np.random.randn(16, 16).astype(np.float64)
    s = (12, 12)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3: 2D complex64 array, padded size, forward norm
    a = (np.random.randn(10, 10) + 1j * np.random.randn(10, 10)).astype(np.complex64)
    s = (15, 15)
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4: 3D complex128 array, transform along last two axes
    a = (np.random.randn(3, 8, 8) + 1j * np.random.randn(3, 8, 8)).astype(np.complex128)
    s = (8, 8)
    axes = (1, 2)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5: 3D float32 array, transform along non-adjacent axes
    a = np.random.randn(4, 12, 12).astype(np.float32)
    s = (6, 6)
    axes = (0, 2)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6: 4D float64 array, transform along last two axes
    a = np.random.randn(2, 2, 8, 8).astype(np.float64)
    s = (8, 8)
    axes = (2, 3)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7: 2D float32 array with negative values, padded, ortho norm
    a = (np.random.randn(5, 5) * 10).astype(np.float32)
    s = (8, 8)
    axes = (-2, -1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8: 3D float64 array with negative values, backward norm
    a = (np.random.randn(3, 10, 10) - 5.0).astype(np.float64)
    s = (10, 10)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9: 4D complex64 array, arbitrary axes
    a = (np.random.randn(2, 5, 2, 5) + 1j * np.random.randn(2, 5, 2, 5)).astype(np.complex64)
    s = (4, 4)
    axes = (1, 3)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10: 2D large float32 array, cropped size, ortho norm
    a = np.random.randn(64, 64).astype(np.float32)
    s = (32, 32)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fft2_1"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_1'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_1'], lib="jax", suffix=1)
