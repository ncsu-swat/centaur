
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default-like parameters
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 array, ortho norm
    a = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(
        np.complex64
    )
    s = [4, 6]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 array, forward norm
    a = np.random.randn(2, 5, 5).astype(np.float64)
    s = [5, 5]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D complex128 array with non-consecutive axes
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex128
    )
    s = [3, 5]
    axes = [0, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: Zero-padding via s larger than input shape
    a = np.random.randn(4, 4).astype(np.float32)
    s = [8, 8]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: Truncation via s smaller than input shape
    a = np.random.randn(10, 12).astype(np.float32)
    s = [6, 6]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D float32 array
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = [4, 5]
    axes = [-2, -1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D array transforming over axes [1, 2]
    a = np.random.randn(2, 6, 8, 3).astype(np.float64)
    s = [6, 8]
    axes = [1, 2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D integer input array (converted/promoted by fft2)
    a = np.random.randint(-10, 10, size=(5, 7), dtype=np.int32)
    s = [5, 7]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: Negative axes indices on 3D array
    a = np.random.randn(4, 6, 8).astype(np.float32)
    s = [4, 6]
    axes = [-3, -2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 5D complex64 array
    a = (np.random.randn(2, 2, 3, 4, 4) + 1j * np.random.randn(2, 2, 3, 4, 4)).astype(
        np.complex64
    )
    s = [4, 4]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fft2_4"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_4'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_4'], lib="jax", suffix=4)
