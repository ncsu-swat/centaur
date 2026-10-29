
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default axes
    a = np.random.randn(4, 4).astype(np.float32)
    s = (4, 4)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 array, ortho normalization
    a = (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(
        np.complex64
    )
    s = (8, 8)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 array with forward normalization
    a = np.random.randn(2, 4, 6).astype(np.float64)
    s = (3, 5)
    axes = (1, 2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D complex128 array, non-adjacent axes
    a = (np.random.randn(3, 5, 4) + 1j * np.random.randn(3, 5, 4)).astype(
        np.complex128
    )
    s = (2, 4)
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D integer array, negative axes indices
    a = np.random.randint(-10, 10, size=(2, 3, 4, 5), dtype=np.int32)
    s = (6, 6)
    axes = (-1, -3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 2D float32 array, zero-padding via s
    a = np.random.randn(4, 4).astype(np.float32)
    s = (10, 5)
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 3D float32 array, truncating via s
    a = np.random.randn(2, 5, 5).astype(np.float32)
    s = (2, 2)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D float32 array with positive axes
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    s = (4, 2)
    axes = (1, 3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D array, reversed axes order
    a = np.random.randn(3, 3).astype(np.float64)
    s = (3, 3)
    axes = (1, 0)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 5D float32 array
    a = np.random.randn(2, 1, 3, 4, 5).astype(np.float32)
    s = (2, 3)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 2D complex64 array, reversed negative axes
    a = (np.random.randn(6, 7) + 1j * np.random.randn(6, 7)).astype(
        np.complex64
    )
    s = (5, 5)
    axes = (-1, -2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

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
