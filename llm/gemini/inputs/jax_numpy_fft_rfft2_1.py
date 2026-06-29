
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfft2_inputs():
    list_of_inputs = []

    # Case 1: Simple 2D float32 array, default axes
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Padding using 's', ortho norm
    a = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (12, 12),
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D float64 array, specifying axes
    a = np.random.randn(3, 16, 16).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (16, 16),
        "axes": (1, 2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Cropping using 's', backward norm
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D array with non-contiguous axes (0, 2)
    a = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 6),
        "axes": (0, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Different size along dimensions in 's'
    a = np.random.randn(8, 8).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (6, 10),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 4D array, specifying inner axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 5),
        "axes": (2, 3),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Odd sizes in 's'
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (3, 5),
        "axes": (0, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Large dimensions
    a = np.random.randn(12, 12).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 16),
        "axes": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 4D array, float64
    a = np.random.randn(2, 2, 8, 8).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfft2_1"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft2_1'.")


check_valid('jax.numpy.fft.rfft2', generated_inputs['jax.numpy.fft.rfft2_1'], lib="jax", suffix=1)
