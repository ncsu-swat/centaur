
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftn_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float32 array, full transform
    a = np.random.randn(4, 4).astype(np.float32)
    s = [4, 4]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D complex64 array, ortho norm
    a = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(np.complex64)
    s = [3, 5]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3: 3D float64 array, partial transform on first two axes, forward norm
    a = np.random.randn(2, 3, 4).astype(np.float64)
    s = [2, 3]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4: 1D float32 array, backward norm
    a = np.random.randn(8).astype(np.float32)
    s = [8]
    axes = [0]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5: 3D complex128 array, full transform, ortho norm
    a = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex128)
    s = [3, 3, 3]
    axes = [0, 1, 2]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6: 4D float32 array, transform on subset of axes, forward norm
    a = np.random.randn(2, 4, 4, 3).astype(np.float32)
    s = [4, 4]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7: 2D float32 array, cropping along transform axes, backward norm
    a = np.random.randn(6, 6).astype(np.float32)
    s = [4, 4]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8: 2D complex64 array, transform on axis 1, ortho norm
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex64)
    s = [8]
    axes = [1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9: 3D float64 array, non-contiguous axes order, forward norm
    a = np.random.randn(3, 4, 5).astype(np.float64)
    s = [5, 3]
    axes = [2, 0]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10: 5D float32 array, small dimensions, backward norm
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    s = [2, 2]
    axes = [1, 3]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 11: 2D float32 array, padding along transform axes, ortho norm
    a = np.random.randn(2, 2).astype(np.float32)
    s = [4, 4]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifftn_1"] = ifftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftn_1'.")


check_valid('jax.numpy.fft.ifftn', generated_inputs['jax.numpy.fft.ifftn_1'], lib="jax", suffix=1)
