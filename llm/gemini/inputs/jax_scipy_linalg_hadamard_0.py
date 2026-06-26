
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hadamard_inputs():
    list_of_inputs = []

    # Input 1: n=1, dtype=int32
    input_dict = {
        "n": 1,
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: n=2, dtype=int64
    input_dict = {
        "n": 2,
        "dtype": np.dtype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n=4, dtype=float32
    input_dict = {
        "n": 4,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: n=8, dtype=float64
    input_dict = {
        "n": 8,
        "dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: n=16, dtype=int16
    input_dict = {
        "n": 16,
        "dtype": np.dtype(np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: n=32, dtype=uint32
    input_dict = {
        "n": 32,
        "dtype": np.dtype(np.uint32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: n=64, dtype=complex64
    input_dict = {
        "n": 64,
        "dtype": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: n=128, dtype=int8
    input_dict = {
        "n": 128,
        "dtype": np.dtype(np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: n=256, dtype=float32
    input_dict = {
        "n": 256,
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: n=512, dtype=int32
    input_dict = {
        "n": 512,
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.hadamard"] = hadamard_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.hadamard' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.hadamard'.")


check_valid('jax.scipy.linalg.hadamard', generated_inputs['jax.scipy.linalg.hadamard'], lib="jax", suffix=0)
