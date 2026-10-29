
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array, forward norm
    a = np.array([-1.0, 2.5, -3.2, 4.8], dtype=np.float32)
    s = (4,)
    axes = (0,)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D array, complex128, backward norm
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex128
    )
    s = (4, 4)
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D array, ortho norm, transform along single axis
    a = np.random.randn(5, 6).astype(np.float64)
    s = (6,)
    axes = (1,)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array, cropped shape s, negative axis
    a = np.random.randn(6, 8, 10).astype(np.float32)
    s = (4, 4)
    axes = (-2, -1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D array, padded shape s, all axes specified
    a = np.random.randn(3, 3, 3).astype(np.complex64)
    s = (5, 6, 7)
    axes = (0, 1, 2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D array, float32, transformed over non-contiguous axes
    a = np.random.randn(2, 4, 3, 5).astype(np.float32)
    s = (3, 4)
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D array, negative axes indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    s = (4, 2)
    axes = (-3, -4)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 1D array, int32 input, backward norm, padded
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    s = (8,)
    axes = (0,)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D array, truncated shape on first axis
    a = (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(
        np.complex64
    )
    s = (3,)
    axes = (0,)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 5D array, transforming a subset of axes
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    s = (2, 3, 4)
    axes = (1, 2, 4)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftn_2"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_2'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_2'], lib="jax", suffix=2)
