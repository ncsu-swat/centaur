
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfftn_inputs():
    list_of_inputs = []

    # Input 1
    a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex64)
    s = [4, 4]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2
    a = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex128)
    s = [8, 8]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3
    a = np.random.randn(3, 3, 3).astype(np.float32)
    s = [3, 3, 4]
    axes = [0, 1, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4
    a = (np.random.randn(10, 6) + 1j * np.random.randn(10, 6)).astype(np.complex64)
    s = [10, 10]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5
    a = (np.random.randn(2, 3, 3, 3) + 1j * np.random.randn(2, 3, 3, 3)).astype(np.complex64)
    s = [3, 4]
    axes = [1, 2]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex64)
    s = [8]
    axes = [1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7
    a = (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(np.complex128)
    s = [4, 6]
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8
    a = np.random.randn(6, 4, 2).astype(np.float64)
    s = [6, 4, 2]
    axes = [0, 1, 2]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9
    a = (np.random.randn(8, 2, 2) + 1j * np.random.randn(8, 2, 2)).astype(np.complex64)
    s = [2, 2]
    axes = [1, 2]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10
    a = (np.random.randn(3, 4, 5, 6) + 1j * np.random.randn(3, 4, 5, 6)).astype(np.complex64)
    s = [4, 5, 10]
    axes = [1, 2, 3]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

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
