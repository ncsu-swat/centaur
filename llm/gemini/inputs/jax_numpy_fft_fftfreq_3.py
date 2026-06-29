
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_fft_fftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 case
    input_dict = {
        "n": 8,
        "d": np.array(1.0, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with different sample spacing
    input_dict = {
        "n": 10,
        "d": np.array(0.5, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Odd n value with float32
    input_dict = {
        "n": 15,
        "d": np.array(0.1, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large n with small spacing
    input_dict = {
        "n": 100,
        "d": np.array(0.01, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative sample spacing (valid mathematically, reverses frequency sign)
    input_dict = {
        "n": 12,
        "d": np.array(-0.25, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large spacing
    input_dict = {
        "n": 6,
        "d": np.array(10.0, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16 dtype
    input_dict = {
        "n": 16,
        "d": np.array(1.0, dtype=np.float16),
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small n
    input_dict = {
        "n": 2,
        "d": np.array(1.0, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large spacing with float64
    input_dict = {
        "n": 50,
        "d": np.array(100.0, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Odd n with negative spacing
    input_dict = {
        "n": 7,
        "d": np.array(-1.5, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftfreq_3"] = jax_numpy_fft_fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftfreq_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftfreq_3'.")


check_valid('jax.numpy.fft.fftfreq', generated_inputs['jax.numpy.fft.fftfreq_3'], lib="jax", suffix=3)
