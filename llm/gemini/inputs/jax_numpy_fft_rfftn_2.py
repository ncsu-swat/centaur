
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_rfftn_inputs():
    list_of_inputs = []

    # Input 1: 2D array, standard parameters
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, ortho norm
    a = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 5),
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, forward norm
    a = np.random.randn(16).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (10,),
        "axes": (0,),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, float64
    a = np.random.randn(3, 3, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (3, 3),
        "axes": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative inputs, different shapes
    a = -np.random.rand(5, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 8),
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large 2D array, forward norm
    a = np.random.randn(32, 32).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (16, 16),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with negative axes
    a = np.random.randn(6, 6, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High dimensional 5D array
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (2, 2, 2),
        "axes": (1, 2, 3),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 array, downscaled sizes
    a = np.random.randn(12, 12).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (6, 6),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, transforming all axes
    a = np.random.randn(5, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (5, 5, 5),
        "axes": (0, 1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfftn_2"] = generate_rfftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftn_2'.")


check_valid('jax.numpy.fft.rfftn', generated_inputs['jax.numpy.fft.rfftn_2'], lib="jax", suffix=2)
