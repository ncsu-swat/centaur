
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifftn_inputs():
    list_of_inputs = []

    # Input 1: 2D array, full transform, default backward norm
    a = np.array([[1.0, 2.0, 5.0, 3.0], [4.0, 1.0, 2.0, 6.0]], dtype=np.float32)
    s = [2, 4]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex array, ortho norm
    a = (
        np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    ).astype(np.complex64)
    s = [4, 4]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 1D array transformed as 1D, forward norm
    a = np.linspace(-5.0, 5.0, 16, dtype=np.float64)
    s = [16]
    axes = [0]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D array, transform along last two axes with padding in s
    a = np.random.randn(3, 4, 5).astype(np.float32)
    s = [8, 8]
    axes = [1, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D complex array, transform along all axes with truncation in s
    a = (
        np.random.randn(6, 6, 6) + 1j * np.random.randn(6, 6, 6)
    ).astype(np.complex128)
    s = [4, 4, 4]
    axes = [0, 1, 2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D array, transform on non-consecutive axes
    a = np.random.randn(2, 5, 3, 4).astype(np.float64)
    s = [2, 4]
    axes = [0, 3]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D array, transform along negative axis index
    a = np.array(
        [[-1.5, 2.5, -3.5], [4.5, -5.5, 6.5]], dtype=np.float32
    )
    s = [3]
    axes = [-1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 3D array, negative axis indices
    a = np.random.randn(4, 8, 16).astype(np.float32)
    s = [8, 16]
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 4D complex array, transform across first three axes
    a = (
        np.random.randn(3, 4, 5, 2) + 1j * np.random.randn(3, 4, 5, 2)
    ).astype(np.complex64)
    s = [3, 4, 5]
    axes = [0, 1, 2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D array with single axis transform, axis 0, padded shape
    a = np.ones((5, 10), dtype=np.float32) * -2.0
    s = [12]
    axes = [0]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.ifftn_1"] = ifftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftn_1'.")


check_valid('jax.numpy.fft.ifftn', generated_inputs['jax.numpy.fft.ifftn_1'], lib="jax", suffix=1)
