
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default-like axes, backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = (-2, -1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D float64 array, ortho norm
    a = np.random.randn(10, 12).astype(np.float64)
    s = [10, 12]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 3D float32 array, forward norm
    a = np.random.randn(4, 6, 8).astype(np.float32)
    s = [6, 8]
    axes = (1, 2)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array with negative axes specification
    a = np.random.randn(5, 7, 9).astype(np.float32)
    s = [5, 7]
    axes = (-3, -2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: Padding in s larger than input shape
    a = np.random.randn(4, 4).astype(np.float32)
    s = [8, 16]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: Truncating in s smaller than input shape
    a = np.random.randn(16, 16).astype(np.float64)
    s = [4, 4]
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D float32 array across arbitrary axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = [2, 5]
    axes = (0, 3)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: Odd dimensions in s
    a = np.random.randn(7, 11).astype(np.float32)
    s = [7, 11]
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: Integer input array converted to real float for FFT
    a = np.random.randint(-10, 10, size=(6, 6)).astype(np.int32)
    s = [6, 6]
    axes = (0, 1)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 5D array
    a = np.random.randn(2, 2, 2, 8, 8).astype(np.float32)
    s = [8, 8]
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: Non-adjacent axes in 3D array
    a = np.random.randn(4, 5, 6).astype(np.float64)
    s = [4, 6]
    axes = (0, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfft2_2"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft2_2'.")


check_valid('jax.numpy.fft.rfft2', generated_inputs['jax.numpy.fft.rfft2_2'], lib="jax", suffix=2)
