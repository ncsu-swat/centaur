
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfft2_inputs():
    list_of_inputs = []

    # Input 1, valid — 2D float32 array, backward norm
    a1 = np.random.randn(8, 8).astype(np.float32)
    input_dict1 = {
        "a": a1,
        "s": [8, 8],
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2, valid — 3D float32 array, ortho norm, padding along axes
    a2 = np.random.randn(4, 6, 6).astype(np.float32)
    input_dict2 = {
        "a": a2,
        "s": [8, 8],
        "axes": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3, valid — float64 array, forward norm, transform along first two axes
    a3 = np.random.randn(5, 5, 3).astype(np.float64)
    input_dict3 = {
        "a": a3,
        "s": [5, 5],
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4, valid — 2D array, cropping with smaller s
    a4 = np.random.randn(10, 10).astype(np.float32)
    input_dict4 = {
        "a": a4,
        "s": [5, 5],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5, valid — 4D array, transform along first and last axes
    a5 = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_dict5 = {
        "a": a5,
        "s": [3, 6],
        "axes": (0, 3),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6, valid — array with negative integers converted to float, backward norm
    a6 = np.random.randint(-10, 11, size=(6, 6)).astype(np.float32)
    input_dict6 = {
        "a": a6,
        "s": [6, 6],
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7, valid — odd s shape dimensions, forward norm
    a7 = np.random.randn(7, 7).astype(np.float32)
    input_dict7 = {
        "a": a7,
        "s": [3, 5],
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8, valid — 3D array with axes (0, 2)
    a8 = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict8 = {
        "a": a8,
        "s": [4, 6],
        "axes": (0, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9, valid — 2D float64 array with larger s shape
    a9 = np.random.randn(12, 12).astype(np.float64)
    input_dict9 = {
        "a": a9,
        "s": [10, 14],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10, valid — 4D array with negative axes indices
    a10 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict10 = {
        "a": a10,
        "s": [3, 4],
        "axes": (-3, -2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
