
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftn_inputs():
    list_of_inputs = []

    # Input 1: 1D real array, backward norm
    a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    s = [4]
    axes = (0,)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex array, forward norm
    a = (
        np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    ).astype(np.complex64)
    s = [4, 4]
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D float array with truncation via s, ortho norm
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    s = [2, 3]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 2D float array with zero-padding via s
    a = np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float32)
    s = [4, 5]
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D float array transformed along a subset of axes (axes 1 and 2)
    a = np.random.randn(2, 6, 8).astype(np.float32)
    s = [6, 8]
    axes = (1, 2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 3D complex array with negative axis indexing
    a = (
        np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)
    ).astype(np.complex128)
    s = [4, 5]
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 3D float array transformed across all 3 axes
    a = np.random.randn(4, 4, 4).astype(np.float32)
    s = [4, 4, 4]
    axes = (0, 1, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D float array transformed along non-adjacent axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    s = [2, 5]
    axes = (0, 3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 4D complex array transformed along all axes with padding
    a = (
        np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)
    ).astype(np.complex64)
    s = [3, 3, 3, 3]
    axes = (0, 1, 2, 3)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D integer array (casted implicitly to float/complex by fft)
    a = np.array([[-5, 10, -15], [20, -25, 30]], dtype=np.int32)
    s = [2, 3]
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 3D array with single axis transform specified via 1-element list/tuple
    a = np.random.randn(5, 7, 9).astype(np.float32)
    s = [7]
    axes = (1,)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftn_3"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_3'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_3'], lib="jax", suffix=3)
