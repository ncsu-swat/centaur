
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D complex64 array with ortho norm
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    s = [6, 6]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3: 3D float64 array, transforming last two axes
    a = np.random.randn(3, 10, 10).astype(np.float64)
    s = [10, 10]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4: 3D complex128 array, transforming first and last axes
    a = (np.random.randn(4, 5, 4) + 1j * np.random.randn(4, 5, 4)).astype(np.complex128)
    s = [4, 4]
    axes = [0, 2]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5: 4D float32 array, custom s size larger than dimensions
    a = np.random.randn(2, 2, 4, 4).astype(np.float32)
    s = [8, 8]
    axes = [2, 3]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6: 2D float32 array, custom s size smaller than dimensions
    a = np.random.randn(10, 10).astype(np.float32)
    s = [5, 5]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7: 3D float32 array with negative values and negative axes indexing
    a = np.random.uniform(-10.0, 10.0, (4, 8, 8)).astype(np.float32)
    s = [8, 8]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8: 4D complex64 array with non-adjacent axes
    a = (np.random.randn(3, 4, 3, 4) + 1j * np.random.randn(3, 4, 3, 4)).astype(np.complex64)
    s = [3, 3]
    axes = [0, 2]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9: Large 2D float64 array
    a = np.random.randn(32, 32).astype(np.float64)
    s = [16, 16]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10: 3D float32 array, rectangular s shape
    a = np.random.randn(5, 5, 5).astype(np.float32)
    s = [3, 7]
    axes = [1, 2]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifft2_3"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_3'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_3'], lib="jax", suffix=3)
