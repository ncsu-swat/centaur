
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfft2_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, standard axes
    a = np.random.randn(4, 6).astype(np.float32)
    s = (4, 6)
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 2: 3D array, float64, ortho normalization
    a = np.random.randn(2, 8, 8).astype(np.float64)
    s = (8, 8)
    axes = [1, 2]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 3: 2D array, forward normalization, odd output sizes
    a = np.random.randn(5, 7).astype(np.float32)
    s = (3, 5)
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 4: 4D array, float32, padding via s
    a = np.random.randn(2, 3, 4, 4).astype(np.float32)
    s = (8, 8)
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 5: 3D integer array
    a = np.random.randint(-10, 10, size=(3, 4, 5), dtype=np.int32)
    s = (3, 4)
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 6: 2D array, swapped axes order
    a = np.random.randn(4, 5).astype(np.float64)
    s = (5, 4)
    axes = [-1, -2]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 7: 5D array, non-adjacent axes
    a = np.random.randn(1, 2, 3, 2, 4).astype(np.float32)
    s = (3, 4)
    axes = [2, 4]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 8: 2D array, cropping via s
    a = np.random.randn(10, 12).astype(np.float32)
    s = (6, 8)
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 9: 3D array, negative indexing with non-adjacent axes
    a = np.random.randn(4, 2, 6).astype(np.float32)
    s = (4, 6)
    axes = [-3, -1]
    norm = "forward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 10: 2D small array with s=(1, 1)
    a = np.random.randn(2, 2).astype(np.float32)
    s = (1, 1)
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    # Input 11: 4D int64 array with negative values
    a = np.random.randint(-5, 5, size=(3, 2, 5, 2), dtype=np.int64)
    s = (3, 5)
    axes = [0, 2]
    norm = "backward"
    list_of_inputs.append(
        copy.deepcopy({"a": a, "s": s, "axes": axes, "norm": norm})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfft2_3"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft2_3'.")


check_valid('jax.numpy.fft.rfft2', generated_inputs['jax.numpy.fft.rfft2_3'], lib="jax", suffix=3)
