
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ihfft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default axis, backward norm
    a = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, axis 0, ortho norm
    a = np.random.randn(16).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 16,
        "axis": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, forward norm
    a = np.random.randn(4, 8).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis 0, backward norm, n larger than dimension
    a = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 10,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, axis 2, ortho norm, n smaller than dimension
    a = np.random.randn(2, 3, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": 2,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 1, forward norm
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64 array, axis -2, backward norm
    a = np.random.randn(2, 2, 4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": -2,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, axis 0, ortho norm, odd n
    a = np.random.randn(7).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 7,
        "axis": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, negative axis, forward norm
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": -1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array, axis 0, backward norm, odd n
    a = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 3,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ihfft"] = ihfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ihfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ihfft'.")


check_valid('jax.numpy.fft.ihfft', generated_inputs['jax.numpy.fft.ihfft'], lib="jax", suffix=0)
