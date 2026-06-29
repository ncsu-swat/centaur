
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft2_inputs():
    list_of_inputs = []

    # Case 1: 2D float32, same size
    a = np.random.randn(8, 8).astype(np.float32)
    s = (8, 8)
    axes = [-2, -1]
    norm = 'backward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 2: 2D float64, ortho norm
    a = np.random.randn(16, 16).astype(np.float64)
    s = (16, 16)
    axes = [0, 1]
    norm = 'ortho'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 3: 3D complex64, different axes
    a = (np.random.randn(4, 8, 8) + 1j * np.random.randn(4, 8, 8)).astype(np.complex64)
    s = (8, 8)
    axes = [1, 2]
    norm = 'forward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 4: 4D complex128
    a = (np.random.randn(2, 3, 8, 8) + 1j * np.random.randn(2, 3, 8, 8)).astype(np.complex128)
    s = (8, 8)
    axes = [-2, -1]
    norm = 'backward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 5: 2D int32, padded output size
    a = np.random.randint(-10, 10, size=(10, 10)).astype(np.int32)
    s = (12, 12)
    axes = [0, 1]
    norm = 'ortho'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 6: 3D float32, axes 0 and 2
    a = np.random.randn(5, 5, 5).astype(np.float32)
    s = (5, 5)
    axes = [0, 2]
    norm = 'forward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 7: 2D complex64, cropped output size
    a = (np.random.randn(16, 32) + 1j * np.random.randn(16, 32)).astype(np.complex64)
    s = (8, 16)
    axes = [0, 1]
    norm = 'backward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 8: 3D float64, ortho norm
    a = np.random.randn(3, 6, 6).astype(np.float64)
    s = (6, 6)
    axes = [-2, -1]
    norm = 'ortho'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 9: 4D float32, inner axes
    a = np.random.randn(2, 2, 4, 4).astype(np.float32)
    s = (4, 4)
    axes = [2, 3]
    norm = 'forward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Case 10: 2D int16, pad output size
    a = np.random.randint(-5, 5, size=(3, 3)).astype(np.int16)
    s = (5, 5)
    axes = [0, 1]
    norm = 'backward'
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fft2_3"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_3'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_3'], lib="jax", suffix=3)
