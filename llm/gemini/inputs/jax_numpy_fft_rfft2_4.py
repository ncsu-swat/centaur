
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfft2_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(8, 8).astype(np.float32)
    s = [8, 8]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 2
    a = np.random.randn(10, 10).astype(np.float64)
    s = [10, 10]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 3
    a = np.random.randn(4, 6, 8).astype(np.float32)
    s = [6, 8]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 4
    a = np.random.randn(5, 5).astype(np.float32)
    s = [8, 8]
    axes = [0, 1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 5
    a = np.random.randint(-10, 10, size=(6, 6)).astype(np.int32)
    s = [4, 4]
    axes = [0, 1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 6
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    s = [3, 4]
    axes = [1, 2]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 7
    a = np.random.randn(12, 12).astype(np.float32)
    s = [10, 15]
    axes = [-2, -1]
    norm = "backward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 8
    a = np.random.randn(3, 3, 3).astype(np.float32)
    s = [3, 3]
    axes = [0, 2]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 9
    a = np.random.randn(7, 9).astype(np.float64)
    s = [5, 12]
    axes = [0, 1]
    norm = "forward"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    # Input 10
    a = np.random.randn(8, 8).astype(np.float32)
    s = [16, 16]
    axes = [-2, -1]
    norm = "ortho"
    list_of_inputs.append({"a": a, "s": s, "axes": axes, "norm": norm})

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfft2_4"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft2_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft2_4'.")


check_valid('jax.numpy.fft.rfft2', generated_inputs['jax.numpy.fft.rfft2_4'], lib="jax", suffix=4)
