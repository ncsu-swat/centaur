
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_fft_ifft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D complex64, standard backward norm
    a = (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(np.complex64)
    s = [8, 8]
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2: 2D float32, ortho norm, different sizes
    a = np.random.randn(10, 12).astype(np.float32)
    s = [5, 6]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3: 3D complex128, forward norm, axes over last two dimensions
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex128)
    s = [4, 4]
    axes = (1, 2)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4: 3D float64, non-adjacent axes
    a = np.random.randn(2, 3, 4).astype(np.float64)
    s = [2, 5]
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5: 2D complex64 with negative and large values, ortho norm
    a = (np.random.randn(6, 6) * 10.0 + 1j * np.random.randn(6, 6) * 10.0).astype(np.complex64)
    s = [6, 6]
    axes = (-2, -1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6: 3D float32, axes over first two dimensions
    a = np.random.randn(4, 4, 4).astype(np.float32)
    s = [3, 3]
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7: 4D float64, backward norm, axes on (2, 3)
    a = np.random.randn(5, 5, 5, 5).astype(np.float64)
    s = [5, 5]
    axes = (2, 3)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8: 2D complex128, ortho norm, up-sampling via larger s
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex128)
    s = [8, 16]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9: 2D float32, zero array, forward norm
    a = np.zeros((3, 3), dtype=np.float32)
    s = [3, 3]
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10: 4D float32, non-adjacent negative axes indices
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    s = [4, 4]
    axes = (-3, -1)
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifft2_1"] = jax_numpy_fft_ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_1'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_1'], lib="jax", suffix=1)
