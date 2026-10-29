
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifft2_inputs():
    list_of_inputs = []

    # Input 1: 2D complex64, default-like axes, backward norm
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    s = [4, 4]
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D float32, ortho norm, upsampling
    a = np.random.randn(3, 5).astype(np.float32)
    s = [6, 10]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D complex128, forward norm, downsampling
    a = (np.random.randn(2, 8, 8) + 1j * np.random.randn(2, 8, 8)).astype(
        np.complex128
    )
    s = [4, 4]
    axes = (1, 2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D float64, axes=(0, 1), backward norm
    a = np.random.randn(4, 6, 3).astype(np.float64)
    s = [4, 6]
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D complex64, axes=(1, 3), ortho norm
    a = (np.random.randn(2, 4, 3, 5) + 1j * np.random.randn(2, 4, 3, 5)).astype(
        np.complex64
    )
    s = [4, 5]
    axes = (1, 3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 2D int32 array with negative values
    a = np.array([[-5, 2, -1], [4, -8, 3]], dtype=np.int32)
    s = [2, 3]
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 3D float32 with negative axes indices (-3, -1)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    s = [3, 5]
    axes = (-3, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D float64 with larger upsampling along axes (2, 3)
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    s = [8, 8]
    axes = (2, 3)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D complex64 with zeros and ones, swapped axes (1, 0)
    a = np.zeros((4, 4), dtype=np.complex64)
    a[0, 0] = 1.0 + 2.0j
    s = [4, 4]
    axes = (1, 0)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 3D complex128, non-square output shape with downsampling and upsampling
    a = (np.random.randn(5, 6, 7) + 1j * np.random.randn(5, 6, 7)).astype(
        np.complex128
    )
    s = [3, 10]
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.ifft2_1"] = ifft2_inputs()

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
