
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default-like axes, backward norm
    a = np.random.randn(4, 4).astype(np.float32)
    s = (4, 4)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 array with upsampling, ortho norm
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(
        np.complex64
    )
    s = (8, 16)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 array, forward norm
    a = np.random.randn(2, 6, 6).astype(np.float64)
    s = (5, 5)
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D complex128 array, non-adjacent axes
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex128
    )
    s = (6, 10)
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D float32 array, axes=(1, 3), ortho norm
    a = np.random.randn(2, 5, 3, 7).astype(np.float32)
    s = (4, 8)
    axes = (1, 3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 2D float32 array with inverted axis order
    a = np.random.randn(6, 4).astype(np.float32)
    s = (3, 5)
    axes = (-1, -2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 3D int32 array (converted internally), axes=(0, 1)
    a = np.random.randint(-10, 10, size=(4, 4, 2)).astype(np.int32)
    s = (4, 4)
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D complex64 array with downsampling
    a = (
        np.random.randn(2, 2, 8, 8) + 1j * np.random.randn(2, 2, 8, 8)
    ).astype(np.complex64)
    s = (4, 4)
    axes = (-2, -1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D float32 array, large non-square shape
    a = np.random.randn(12, 18).astype(np.float32)
    s = (16, 20)
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 5D float32 array, arbitrary axes
    a = np.random.randn(2, 3, 4, 3, 5).astype(np.float32)
    s = (4, 6)
    axes = (2, 4)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 3D float64 array with negative axes indices
    a = np.random.randn(3, 4, 5).astype(np.float64)
    s = (6, 8)
    axes = (-3, -1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.ifft2_2"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_2'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_2'], lib="jax", suffix=2)
