
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array, default-like parameters
    a = np.random.randn(8, 8).astype(np.float32)
    s = (8, 8)
    axes = [-2, -1]
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, ortho norm, different sizes for s
    a = np.random.randn(10, 12).astype(np.float64)
    s = (12, 10)
    axes = [0, 1]
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array, transforming last two axes, forward norm
    a = np.random.randn(3, 16, 16).astype(np.float32)
    s = (16, 16)
    axes = [-2, -1]
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, transforming first two axes with cropping
    a = np.random.randn(5, 5, 5).astype(np.float32)
    s = (4, 4)
    axes = [0, 1]
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float64 array, transforming intermediate axes with zero padding
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    s = (6, 6)
    axes = [1, 2]
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 2D float32 array, half size transform
    a = np.random.randn(100, 100).astype(np.float32)
    s = (50, 50)
    axes = [0, 1]
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array with negative/non-adjacent axes
    a = np.random.randn(4, 4, 4).astype(np.float32)
    s = (8, 2)
    axes = [-3, -1]
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float64 array with minimal size
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    s = (3, 3)
    axes = [-2, -1]
    norm = "ortho"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, mismatched padding and cropping
    a = np.random.randn(12, 12).astype(np.float32)
    s = (10, 15)
    axes = [0, 1]
    norm = "forward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, non-adjacent positive axes
    a = np.random.randn(8, 10, 12).astype(np.float32)
    s = (8, 10)
    axes = [0, 2]
    norm = "backward"
    input_dict = {"a": a, "s": s, "axes": axes, "norm": norm}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
