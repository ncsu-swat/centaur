
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def irfftn_inputs():
    list_of_inputs = []

    # Input 1: 3D complex array, default-like transform across all axes
    # Last axis of rfft output of size 8 would be 8 // 2 + 1 = 5
    a = (
        np.random.randn(4, 4, 5).astype(np.float32)
        + 1j * np.random.randn(4, 4, 5).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [4, 4, 8],
        "axes": [0, 1, 2],
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, ortho norm, odd last dimension size
    # For output size 7 on last axis, input size is 7 // 2 + 1 = 4
    a = (
        np.random.randn(6, 4).astype(np.float64)
        + 1j * np.random.randn(6, 4).astype(np.float64)
    ).astype(np.complex128)
    input_dict = {"a": a, "s": [6, 7], "axes": [0, 1], "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D array, forward norm, transform along a subset of axes (axes 1 and 3)
    # output size along axis 1: 5, axis 3: 6 -> input size along axis 3: 6 // 2 + 1 = 4
    a = (
        np.random.randn(2, 5, 3, 4).astype(np.float32)
        + 1j * np.random.randn(2, 5, 3, 4).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"a": a, "s": [5, 6], "axes": [1, 3], "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D equivalent via irfftn, 1 axis specified
    a = (
        np.random.randn(9).astype(np.float32)
        + 1j * np.random.randn(9).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"a": a, "s": [16], "axes": [0], "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative axes specification
    a = (
        np.random.randn(3, 4, 6).astype(np.float64)
        + 1j * np.random.randn(3, 4, 6).astype(np.float64)
    ).astype(np.complex128)
    input_dict = {"a": a, "s": [4, 10], "axes": [-2, -1], "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Real-valued input array (irfftn accepts real inputs)
    a = np.random.randn(4, 5, 3).astype(np.float32)
    input_dict = {"a": a, "s": [4, 5, 4], "axes": [0, 1, 2], "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with single axis transformation (axis 0)
    # Last axis in 'axes' is axis 0, so output shape along axis 0 is s[0]=6
    a = (
        np.random.randn(4, 8, 8).astype(np.float32)
        + 1j * np.random.randn(4, 8, 8).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"a": a, "s": [6], "axes": [0], "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D array, transformed over 3 non-consecutive axes
    a = (
        np.random.randn(2, 3, 2, 4, 5).astype(np.float32)
        + 1j * np.random.randn(2, 3, 2, 4, 5).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [2, 4, 8],
        "axes": [0, 3, 4],
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-standard axes order (transposed FFT axes)
    a = (
        np.random.randn(4, 6).astype(np.float64)
        + 1j * np.random.randn(4, 6).astype(np.float64)
    ).astype(np.complex128)
    input_dict = {"a": a, "s": [10, 4], "axes": [1, 0], "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero-filled input with negative values, complex64
    a = (
        np.array([[-1.5, 2.0, -0.5], [0.0, -1.0, 3.2]], dtype=np.float32)
        + 1j * np.array([[0.5, -2.5, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    ).astype(np.complex64)
    input_dict = {"a": a, "s": [2, 4], "axes": [0, 1], "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.irfftn_1"] = irfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfftn_1'.")


check_valid('jax.numpy.fft.irfftn', generated_inputs['jax.numpy.fft.irfftn_1'], lib="jax", suffix=1)
