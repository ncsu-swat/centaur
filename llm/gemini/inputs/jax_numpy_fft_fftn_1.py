
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array with backward norm
    a = np.random.randn(8).astype(np.float32)
    s = [8]
    axes = [0]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D array with ortho norm
    a = np.random.randn(4, 6).astype(np.float32)
    s = [4, 6]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D array with negative axis indexing and forward norm
    a = np.random.randn(5, 5).astype(np.float64)
    s = [3, 4]
    axes = [-2, -1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 2D complex array transforming along a single axis
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(
        np.complex64
    )
    s = [6]
    axes = [1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D array transforming across all axes
    a = np.random.randn(3, 4, 5).astype(np.float32)
    s = [3, 4, 5]
    axes = [0, 1, 2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 3D complex128 array transforming a non-contiguous subset of axes
    a = (np.random.randn(4, 3, 6) + 1j * np.random.randn(4, 3, 6)).astype(
        np.complex128
    )
    s = [4, 6]
    axes = [0, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 4D array transforming across all dimensions
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    s = [2, 3, 2, 4]
    axes = [0, 1, 2, 3]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 4D float64 array with a negative axis index
    a = np.random.randn(2, 4, 3, 5).astype(np.float64)
    s = [6]
    axes = [-3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 2D integer array with negative values
    a = np.random.randint(-10, 10, size=(4, 4), dtype=np.int32)
    s = [4, 4]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 3D array with padding/truncating shape parameter
    a = np.random.randn(4, 4, 4).astype(np.float32)
    s = [2, 8]
    axes = [1, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 5D array transforming a 2D slice
    a = np.random.randn(2, 3, 2, 4, 2).astype(np.float32)
    s = [3, 4]
    axes = [1, 3]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftn_1"] = fftn_inputs()

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
