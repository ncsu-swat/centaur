
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifftn_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard backward normalization
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = (4,)
    axes = (0,)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 2D complex array, ortho normalization
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    s = (4, 4)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D float array with size modification (cropping)
    a = np.random.randn(5, 5).astype(np.float64)
    s = (2, 3)
    axes = (0, 1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 3D float array transformed along a subset of axes (1, 2)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    s = (4, 5)
    axes = (1, 2)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D complex128 array transformed along all axes
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex128
    )
    s = (2, 3, 4)
    axes = (0, 1, 2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 4D float array with negative axis indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = (4, 5)
    axes = (-2, -1)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 2D array, 1D transform with padding along axis 1
    a = np.random.randn(3, 3).astype(np.float32)
    s = (6,)
    axes = (1,)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 3D array transformed along non-consecutive axes (0, 2)
    a = np.random.randn(4, 5, 6).astype(np.float64)
    s = (2, 3)
    axes = (0, 2)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 1D complex array with negative values
    a = np.array([-1.5 + 2.0j, -3.0 - 0.5j, 4.2 - 1.1j], dtype=np.complex64)
    s = (6,)
    axes = (-1,)
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 4D array fully transformed with small dimensions
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    s = (2, 2, 2, 2)
    axes = (0, 1, 2, 3)
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 2D array cropped down to 1x1
    a = np.random.randn(4, 4).astype(np.float32)
    s = (1, 1)
    axes = (0, 1)
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

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
