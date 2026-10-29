
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic even n with standard float32
    input_dict = {
        "n": 8,
        "d": np.array(1.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd n with float64
    input_dict = {
        "n": 9,
        "d": np.array(0.5, dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small n (n=1)
    input_dict = {
        "n": 1,
        "d": np.array(2.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large even n
    input_dict = {
        "n": 1024,
        "d": np.array(0.01, dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative sample spacing
    input_dict = {
        "n": 16,
        "d": np.array(-0.25, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D tensor for d (shape (1,))
    input_dict = {
        "n": 32,
        "d": np.array([1.5], dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D broadcastable tensor for d (shape (1, 1))
    input_dict = {
        "n": 10,
        "d": np.array([[0.1]], dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer tensor d
    input_dict = {
        "n": 50,
        "d": np.array(3, dtype=np.int32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large odd n with small sample spacing
    input_dict = {
        "n": 501,
        "d": np.array(1e-4, dtype=np.float64),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: n=2 minimal non-trivial even length
    input_dict = {
        "n": 2,
        "d": np.array(10.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
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
