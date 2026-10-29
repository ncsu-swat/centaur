
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic even n, default-like parameters
    input_dict = {"n": 8, "d": 1.0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd n, float64 dtype
    input_dict = {"n": 9, "d": 0.5, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n = 1 minimal positive length
    input_dict = {"n": 1, "d": 2.0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: n = 2 small even length
    input_dict = {"n": 2, "d": 0.1, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large n, small sample spacing
    input_dict = {"n": 1024, "d": 0.001, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative sample spacing
    input_dict = {"n": 16, "d": -1.0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Medium odd n with negative spacing
    input_dict = {"n": 25, "d": -0.25, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Power of 2 with float16 dtype
    input_dict = {"n": 64, "d": 1.5, "dtype": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large odd length
    input_dict = {"n": 513, "d": 10.0, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero window length
    input_dict = {"n": 0, "d": 1.0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex dtype output
    input_dict = {"n": 128, "d": 0.05, "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfftfreq_1"] = rfftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftfreq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftfreq_1'.")


check_valid('jax.numpy.fft.rfftfreq', generated_inputs['jax.numpy.fft.rfftfreq_1'], lib="jax", suffix=1)
