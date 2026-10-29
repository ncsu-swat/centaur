
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like parameters
    a = np.random.randn(8).astype(np.float32)
    s = (8,)
    axes = (0,)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D array, ortho norm, float64
    a = np.random.randn(4, 6).astype(np.float64)
    s = (4, 6)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D array, forward norm
    a = np.random.randn(2, 3, 4).astype(np.float32)
    s = (2, 3, 4)
    axes = (0, 1, 2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array, padding via s, negative axes
    a = np.random.randn(2, 4, 5).astype(np.float32)
    s = (4, 6)
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D array, truncation via s
    a = np.random.randn(6, 8, 10).astype(np.float32)
    s = (3, 4, 5)
    axes = (0, 1, 2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D array, transform along non-consecutive axes
    a = np.random.randn(3, 4, 5, 6).astype(np.float32)
    s = (3, 6)
    axes = (0, 3)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D array, transform along 3 axes
    a = np.random.randn(2, 5, 4, 3).astype(np.float64)
    s = (5, 4, 3)
    axes = (1, 2, 3)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 2D array, reordered axes
    a = np.random.randn(8, 12).astype(np.float32)
    s = (12, 8)
    axes = (1, 0)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 5D array, float32, transformed over 2 axes with negative values
    a = -np.abs(np.random.randn(2, 2, 3, 4, 5).astype(np.float32))
    s = (3, 4)
    axes = (2, 3)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 1D array, large size with padding, forward norm
    a = np.linspace(-10.0, 10.0, 16, dtype=np.float64)
    s = (32,)
    axes = (0,)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 3D array, integer-like float values, all axes
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12.0
    s = (2, 3, 4)
    axes = (-3, -2, -1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfftn_2"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftn_2'.")


check_valid('jax.numpy.fft.rfftn', generated_inputs['jax.numpy.fft.rfftn_2'], lib="jax", suffix=2)
