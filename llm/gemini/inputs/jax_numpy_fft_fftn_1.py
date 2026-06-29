
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_fft_fftn_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, standard case
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D float64 array, ortho normalization
    a = np.random.randn(5, 5).astype(np.float64)
    s = [4, 4]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 3: 3D complex64 array, forward normalization, transform on subset of axes
    a = (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(np.complex64)
    s = [2, 2]
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 4: 1D float32 array, negative values, single element lists for s and axes
    a = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    s = [4]
    axes = [0]
    norm = "backward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 5: 3D float32 array, transform along all axes, larger s
    a = np.random.randn(3, 3, 3).astype(np.float32)
    s = [4, 4, 4]
    axes = [0, 1, 2]
    norm = "ortho"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 6: 4D complex128 array, transform on two axes
    a = (np.random.randn(2, 3, 4, 2) + 1j * np.random.randn(2, 3, 4, 2)).astype(np.complex128)
    s = [3, 2]
    axes = [1, 3]
    norm = "forward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 7: 2D float64 array with zero padding (s larger than input shape)
    a = np.random.randn(3, 3).astype(np.float64)
    s = [6, 6]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 8: 3D float32 array, non-contiguous axes order
    a = np.random.randn(4, 5, 6).astype(np.float32)
    s = [3, 3]
    axes = [2, 0]
    norm = "ortho"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 9: 2D complex64 array, truncation (s smaller than input shape)
    a = (np.random.randn(10, 10) + 1j * np.random.randn(10, 10)).astype(np.complex64)
    s = [5, 5]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    # Input 10: 4D float32 array, transform on 3 axes
    a = np.random.randn(2, 3, 3, 2).astype(np.float32)
    s = [4, 4, 4]
    axes = [0, 1, 2]
    norm = "backward"
    list_of_inputs.append({"a": copy.deepcopy(a), "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftn_1"] = jax_numpy_fft_fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_1'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_1'], lib="jax", suffix=1)
