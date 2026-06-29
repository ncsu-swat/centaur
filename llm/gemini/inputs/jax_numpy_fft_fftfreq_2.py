
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftfreq_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"n": 8, "d": 1, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"n": 16, "d": 2, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"n": 5, "d": 1, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"n": 10, "d": -1, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"n": 128, "d": 3, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"n": 256, "d": 10, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"n": 3, "d": 5, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"n": 1024, "d": 1, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"n": 7, "d": -2, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"n": 50, "d": 4, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftfreq_2"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftfreq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftfreq_2'.")


check_valid('jax.numpy.fft.fftfreq', generated_inputs['jax.numpy.fft.fftfreq_2'], lib="jax", suffix=2)
