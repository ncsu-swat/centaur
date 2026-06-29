
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfftn_inputs():
    list_of_inputs = []

    # Input 1: 2D array, standard sizes
    list_of_inputs.append({
        "a": np.random.randn(8, 8).astype(np.float32),
        "s": [8, 8],
        "axes": [0, 1],
        "norm": "backward"
    })

    # Input 2: 2D array, ortho normalization, different s
    list_of_inputs.append({
        "a": np.random.randn(10, 10).astype(np.float32),
        "s": [10, 5],
        "axes": [0, 1],
        "norm": "ortho"
    })

    # Input 3: 3D array, float64, forward normalization
    list_of_inputs.append({
        "a": np.random.randn(4, 6, 8).astype(np.float64),
        "s": [4, 6, 8],
        "axes": [0, 1, 2],
        "norm": "forward"
    })

    # Input 4: 1D array
    list_of_inputs.append({
        "a": np.random.randn(16).astype(np.float32),
        "s": [16],
        "axes": [0],
        "norm": "backward"
    })

    # Input 5: 4D array, subset of axes
    list_of_inputs.append({
        "a": np.random.randn(3, 4, 5, 6).astype(np.float32),
        "s": [3, 4],
        "axes": [0, 1],
        "norm": "ortho"
    })

    # Input 6: 2D array, padding (s larger than dimension size)
    list_of_inputs.append({
        "a": np.random.randn(5, 5).astype(np.float32),
        "s": [8, 8],
        "axes": [0, 1],
        "norm": "backward"
    })

    # Input 7: 3D array, subset of axes with cropping (s smaller than dimension size)
    list_of_inputs.append({
        "a": np.random.randn(10, 10, 10).astype(np.float64),
        "s": [5, 5],
        "axes": [1, 2],
        "norm": "forward"
    })

    # Input 8: 4D array, transforming all axes
    list_of_inputs.append({
        "a": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "s": [2, 2, 2, 2],
        "axes": [0, 1, 2, 3],
        "norm": "backward"
    })

    # Input 9: 3D array, disjoint axes
    list_of_inputs.append({
        "a": np.random.randn(3, 3, 3).astype(np.float32),
        "s": [3, 2],
        "axes": [0, 2],
        "norm": "ortho"
    })

    # Input 10: 2D array, using negative axes indices
    list_of_inputs.append({
        "a": np.random.randn(12, 12).astype(np.float32),
        "s": [6, 6],
        "axes": [-2, -1],
        "norm": "backward"
    })

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfftn_1"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftn_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftn_1'.")


check_valid('jax.numpy.fft.rfftn', generated_inputs['jax.numpy.fft.rfftn_1'], lib="jax", suffix=1)
