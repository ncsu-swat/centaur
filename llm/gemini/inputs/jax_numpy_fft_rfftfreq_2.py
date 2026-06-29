
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values, float32
    input_dict = {
        "n": 10,
        "d": np.array(1.0, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd n, float64
    input_dict = {
        "n": 15,
        "d": np.array(0.5, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger n, smaller spacing, float32
    input_dict = {
        "n": 128,
        "d": np.array(0.01, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with 1 element for spacing, float64
    input_dict = {
        "n": 256,
        "d": np.array([2.0], dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Minimal window size n=1, float32
    input_dict = {
        "n": 1,
        "d": np.array(0.1, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative spacing, float64
    input_dict = {
        "n": 100,
        "d": np.array(-0.5, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large spacing, float32
    input_dict = {
        "n": 64,
        "d": np.array(100.0, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small odd n, 1D tensor spacing, float32
    input_dict = {
        "n": 5,
        "d": np.array([0.001], dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative spacing and float64 dtype
    input_dict = {
        "n": 50,
        "d": np.array(-1.25, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Power of 2 window size, float32
    input_dict = {
        "n": 1024,
        "d": np.array(1.5, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfftfreq_2"] = rfftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftfreq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftfreq_2'.")


check_valid('jax.numpy.fft.rfftfreq', generated_inputs['jax.numpy.fft.rfftfreq_2'], lib="jax", suffix=2)
