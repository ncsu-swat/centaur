
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftn_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, transform over all axes
    a = np.random.randn(4, 4).astype(np.float32)
    s = (4, 4)
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 array, truncated/padded transform with ortho norm
    a = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(
        np.complex64
    )
    s = (2, 3)
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 array, enlarged transform with forward norm
    a = np.random.randn(6, 6, 6).astype(np.float64)
    s = (8, 8, 8)
    axes = [0, 1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 1D float32 array
    a = np.random.randn(8).astype(np.float32)
    s = (10,)
    axes = [0]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D complex128 array, subset of axes
    a = (np.random.randn(2, 5, 3, 7) + 1j * np.random.randn(2, 5, 3, 7)).astype(
        np.complex128
    )
    s = (2, 4)
    axes = [1, 3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 3D int32 array with negative values
    a = np.random.randint(-10, 10, size=(4, 5, 6)).astype(np.int32)
    s = (3, 3)
    axes = [0, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D float32 array, transform along last axis using negative index
    a = np.random.randn(5, 8).astype(np.float32)
    s = (6,)
    axes = [-1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 3D float32 array with negative axes
    a = (np.random.randn(3, 4, 5) - 2.0).astype(np.float32)
    s = (4, 2)
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 4D float64 array, non-contiguous axes
    a = np.random.randn(2, 4, 3, 5).astype(np.float64)
    s = (2, 3, 2)
    axes = [0, 1, 3]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 5D complex64 array, transform across first 4 axes
    a = (
        np.random.randn(2, 2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2, 2)
    ).astype(np.complex64)
    s = (2, 2, 2, 2)
    axes = [0, 1, 2, 3]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftn_4"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_4'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_4'], lib="jax", suffix=4)
