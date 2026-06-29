
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftn_inputs():
    list_of_inputs = []

    # Input 1, 2D float32, matching shape
    a = np.random.randn(4, 4).astype(np.float32)
    s = (4, 4)
    axes = (0, 1)
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, 2D complex64, up-sampling
    a = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(np.complex64)
    s = (4, 6)
    axes = (0, 1)
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, 3D float64, down-sampling and up-sampling
    a = np.random.randn(2, 3, 4).astype(np.float64)
    s = (3, 3, 3)
    axes = (0, 1, 2)
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, 1D complex128
    a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex128)
    s = (10,)
    axes = (0,)
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, 3D float32, transforming subset of axes
    a = np.random.randn(3, 4, 5).astype(np.float32)
    s = (2, 6)
    axes = (0, 2)
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, 4D complex64, transforming all axes
    a = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    s = (3, 3, 3, 3)
    axes = (0, 1, 2, 3)
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, 2D float64, down-sampling
    a = np.random.randn(5, 5).astype(np.float64)
    s = (4, 4)
    axes = (0, 1)
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, 3D complex128, transforming subset of axes with different sizes
    a = (np.random.randn(2, 4, 6) + 1j * np.random.randn(2, 4, 6)).astype(np.complex128)
    s = (3, 5)
    axes = (1, 2)
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, 1D float32, down-sampling
    a = np.random.randn(16).astype(np.float32)
    s = (8,)
    axes = (0,)
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, 2D complex64, permuted axes
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
    s = (6, 5)
    axes = (1, 0)
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifftn_2"] = ifftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftn_2'.")


check_valid('jax.numpy.fft.ifftn', generated_inputs['jax.numpy.fft.ifftn_2'], lib="jax", suffix=2)
