
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic case with typical parameters
    input_dict = {
        "n": 8,
        "d": 1.0,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd number for n, different float64 dtype
    input_dict = {
        "n": 9,
        "d": 0.5,
        "dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger n, smaller spacing d
    input_dict = {
        "n": 1024,
        "d": 0.001,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative spacing d
    input_dict = {
        "n": 16,
        "d": -0.1,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Minimal positive size n=1
    input_dict = {
        "n": 1,
        "d": 10.0,
        "dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large spacing d
    input_dict = {
        "n": 32,
        "d": 1000.0,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Size 0 (empty FFT)
    input_dict = {
        "n": 0,
        "d": 1.0,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-resolution grid
    input_dict = {
        "n": 5000,
        "d": 2.5,
        "dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small odd n with custom float64 dtype
    input_dict = {
        "n": 3,
        "d": 0.25,
        "dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Standard power of 2 size with small d
    input_dict = {
        "n": 256,
        "d": 0.05,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftfreq_1"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftfreq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftfreq_1'.")


check_valid('jax.numpy.fft.fftfreq', generated_inputs['jax.numpy.fft.fftfreq_1'], lib="jax", suffix=1)
