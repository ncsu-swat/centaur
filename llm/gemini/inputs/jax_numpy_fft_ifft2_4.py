
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifft2_inputs():
    list_of_inputs = []

    # Input 1: 2D complex64 array, default-like axes, backward norm
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    s = (4, 4)
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D float32 array with shape resizing, ortho norm
    a = np.random.randn(2, 3).astype(np.float32)
    s = (4, 6)
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D complex128 array, forward norm
    a = (np.random.randn(3, 8, 8) + 1j * np.random.randn(3, 8, 8)).astype(
        np.complex128
    )
    s = (8, 8)
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D float64 array with negative values, axes on first two dimensions
    a = (np.random.randn(5, 5, 2) * -2.5).astype(np.float64)
    s = (4, 4)
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D complex64 array, ortho norm on axes [2, 3]
    a = (
        np.random.randn(2, 2, 4, 4) + 1j * np.random.randn(2, 2, 4, 4)
    ).astype(np.complex64)
    s = (4, 4)
    axes = [2, 3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D float32 array, non-consecutive axes [0, 2]
    a = np.random.randn(3, 4, 3, 2).astype(np.float32)
    s = (6, 6)
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D integer array, negative axes indices
    a = np.random.randint(-10, 10, size=(6, 6)).astype(np.int32)
    s = (6, 6)
    axes = [-1, -2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 3D complex64 array, truncation via smaller s
    a = (np.random.randn(4, 6, 8) + 1j * np.random.randn(4, 6, 8)).astype(
        np.complex64
    )
    s = (3, 4)
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 5D float32 array, arbitrary axes
    a = np.random.randn(2, 3, 2, 4, 2).astype(np.float32)
    s = (5, 5)
    axes = [1, 3]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D float64 array with zero padding via larger s
    a = np.random.randn(8, 8).astype(np.float64)
    s = (16, 16)
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.ifft2_4"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_4'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_4'], lib="jax", suffix=4)
