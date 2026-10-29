
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic even n with 0D float32 tensor spacing and float32 dtype
    input_dict = {
        "n": 8,
        "d": np.array(1.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd n with 0D float64 tensor spacing and float64 dtype
    input_dict = {
        "n": 9,
        "d": np.array(0.5, dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n=1 edge case with 0D tensor spacing
    input_dict = {
        "n": 1,
        "d": np.array(0.1, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative sample spacing
    input_dict = {
        "n": 16,
        "d": np.array(-0.25, dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large n with integer-typed tensor spacing
    input_dict = {
        "n": 1024,
        "d": np.array(2, dtype=np.int32),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Prime n with 1D single-element tensor
    input_dict = {
        "n": 17,
        "d": np.array([1.5], dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 dtype
    input_dict = {
        "n": 32,
        "d": np.array(0.125, dtype=np.float16),
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 dtype output
    input_dict = {
        "n": 64,
        "d": np.array(0.01, dtype=np.float32),
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 dtype output
    input_dict = {
        "n": 127,
        "d": np.array(10.0, dtype=np.float64),
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-power of 2 even n with negative spacing
    input_dict = {
        "n": 50,
        "d": np.array(-1.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small odd n with 1D tensor
    input_dict = {
        "n": 3,
        "d": np.array([0.05], dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftfreq_3"] = fftfreq_inputs()

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
