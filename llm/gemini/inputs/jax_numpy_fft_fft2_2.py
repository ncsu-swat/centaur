
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array, default-like axes, backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 array with ortho norm
    a = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(
        np.complex64
    )
    s = [4, 6]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 array with forward norm and padding in s
    a = np.random.randn(2, 5, 5).astype(np.float64)
    s = [8, 8]
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array with cropping in s
    a = np.random.randn(3, 10, 12).astype(np.float32)
    s = [4, 6]
    axes = (1, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D array transforming over first and last axes
    a = np.random.randn(6, 4, 8).astype(np.float32)
    s = [6, 8]
    axes = (0, 2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D float32 array, batch dims
    a = np.random.randn(2, 3, 4, 4).astype(np.float32)
    s = [4, 4]
    axes = (2, 3)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D complex128 array with negative axes
    a = (np.random.randn(2, 4, 5, 3) + 1j * np.random.randn(2, 4, 5, 3)).astype(
        np.complex128
    )
    s = [5, 3]
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 2D integer array with negative values
    a = np.random.randint(-10, 10, size=(6, 6), dtype=np.int32)
    s = [6, 6]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 5D float32 tensor
    a = np.random.randn(2, 2, 3, 4, 5).astype(np.float32)
    s = [3, 4]
    axes = (2, 3)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: Non-square transform with transposed axes order
    a = np.random.randn(7, 9).astype(np.float64)
    s = [12, 10]
    axes = (1, 0)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 3D array with negative axis specification
    a = np.random.randn(4, 8, 16).astype(np.float32)
    s = [8, 16]
    axes = (-3, -1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fft2_2"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_2'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_2'], lib="jax", suffix=2)
