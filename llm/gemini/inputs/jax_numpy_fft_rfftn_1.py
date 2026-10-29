
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like parameters
    a = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    s = [4]
    axes = [0]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D array, ortho normalization
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D array, forward normalization, custom truncated shape
    a = np.random.randn(6, 10).astype(np.float64)
    s = [4, 6]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array, padded shape
    a = np.random.randn(2, 3, 4).astype(np.float32)
    s = [4, 5, 6]
    axes = [0, 1, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D array, transform along subset of axes (axes 1 and 2)
    a = np.random.randn(5, 8, 12).astype(np.float32)
    s = [8, 12]
    axes = [1, 2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 3D array, transform along non-consecutive axes (axes 0 and 2)
    a = np.random.randn(4, 6, 8).astype(np.float64)
    s = [4, 8]
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D array, float32, transformed over all axes
    a = np.random.randn(2, 4, 3, 6).astype(np.float32)
    s = [2, 4, 3, 6]
    axes = [0, 1, 2, 3]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D array with negative axis indices
    a = np.random.randn(3, 4, 5, 6).astype(np.float32)
    s = [4, 5]
    axes = [-3, -2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D integer array (casted by FFT)
    a = np.random.randint(-10, 10, size=(6, 8)).astype(np.int32)
    s = [6, 8]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 1D array with padding and forward normalization
    a = np.linspace(-5.0, 5.0, num=10, dtype=np.float64)
    s = [16]
    axes = [0]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 5D array, transforming over last 3 axes
    a = np.random.randn(2, 2, 3, 4, 5).astype(np.float32)
    s = [3, 4, 5]
    axes = [2, 3, 4]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfftn_1"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftn_1'.")


check_valid('jax.numpy.fft.rfftn', generated_inputs['jax.numpy.fft.rfftn_1'], lib="jax", suffix=1)
