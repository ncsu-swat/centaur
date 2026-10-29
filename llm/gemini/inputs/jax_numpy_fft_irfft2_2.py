
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def irfft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, standard axes
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    s = [4, 6]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex128 array, ortho norm
    a = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(
        np.complex128
    )
    s = [8, 8]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D complex64 array, forward norm
    a = (np.random.randn(2, 6, 3) + 1j * np.random.randn(2, 6, 3)).astype(
        np.complex64
    )
    s = [6, 4]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D float64 array, backward norm
    a = np.random.randn(3, 5, 7).astype(np.float64)
    s = [5, 12]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D complex64 array, ortho norm
    a = (
        np.random.randn(2, 3, 4, 3) + 1j * np.random.randn(2, 3, 4, 3)
    ).astype(np.complex64)
    s = [4, 4]
    axes = [2, 3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D float32 array, non-adjacent axes
    a = np.random.randn(5, 2, 6, 3).astype(np.float32)
    s = [5, 8]
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D array, odd output dimensions
    a = (np.random.randn(5, 4) + 1j * np.random.randn(5, 4)).astype(
        np.complex64
    )
    s = [5, 7]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 3D complex64 array, first two axes
    a = (np.random.randn(4, 5, 2) + 1j * np.random.randn(4, 5, 2)).astype(
        np.complex64
    )
    s = [4, 8]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 5D float32 array, negative axes
    a = np.random.randn(2, 3, 4, 5, 3).astype(np.float32)
    s = [5, 4]
    axes = [-2, -1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D array, reversed axes order
    a = (np.random.randn(6, 4) + 1j * np.random.randn(6, 4)).astype(
        np.complex128
    )
    s = [6, 8]
    axes = [1, 0]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 3D complex128 array, negative values, ortho norm
    a = (
        -np.abs(np.random.randn(3, 4, 5))
        - 1j * np.abs(np.random.randn(3, 4, 5))
    ).astype(np.complex128)
    s = [6, 9]
    axes = [-3, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.irfft2_2"] = irfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfft2_2'.")


check_valid('jax.numpy.fft.irfft2', generated_inputs['jax.numpy.fft.irfft2_2'], lib="jax", suffix=2)
