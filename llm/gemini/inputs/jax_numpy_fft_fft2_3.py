
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 standard backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    s = (8, 8)
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex64 with ortho norm
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(
        np.complex64
    )
    s = (6, 6)
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float64 with forward norm along axes [1, 2]
    a = np.random.randn(4, 5, 6).astype(np.float64)
    s = (5, 6)
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D complex128 along non-contiguous axes [0, 2]
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex128
    )
    s = (3, 5)
    axes = [0, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 4D float32 with padding (s larger than slice shape)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = (8, 8)
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D float32 with cropping (s smaller than slice shape)
    a = np.random.randn(3, 4, 6, 8).astype(np.float32)
    s = (2, 3)
    axes = [2, 3]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D int32 array (valid ArrayLike)
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    s = (5, 5)
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 5D float32 array
    a = np.random.randn(2, 2, 3, 4, 5).astype(np.float32)
    s = (3, 4)
    axes = [2, 3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 3D float32 with reversed axes order
    a = np.random.randn(4, 5, 6).astype(np.float32)
    s = (6, 5)
    axes = [2, 1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D float64 with negative values and ortho norm
    a = -np.abs(np.random.randn(7, 9)).astype(np.float64)
    s = (7, 9)
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

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
