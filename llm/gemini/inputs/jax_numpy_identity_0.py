
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def identity_inputs():
    list_of_inputs = []

    # Input 1: 3x3 float32 identity matrix
    list_of_inputs.append({"n": 3, "dtype": np.float32})

    # Input 2: 2x2 int32 identity matrix
    list_of_inputs.append({"n": 2, "dtype": np.int32})

    # Input 3: 5x5 float64 identity matrix
    list_of_inputs.append({"n": 5, "dtype": np.float64})

    # Input 4: 10x10 int64 identity matrix
    list_of_inputs.append({"n": 10, "dtype": np.int64})

    # Input 5: 1x1 complex128 identity matrix
    list_of_inputs.append({"n": 1, "dtype": np.complex128})

    # Input 6: 0x0 float32 empty identity matrix
    list_of_inputs.append({"n": 0, "dtype": np.float32})

    # Input 7: 4x4 boolean identity matrix
    list_of_inputs.append({"n": 4, "dtype": np.bool_})

    # Input 8: 8x8 uint8 identity matrix
    list_of_inputs.append({"n": 8, "dtype": np.uint8})

    # Input 9: 15x15 float16 identity matrix
    list_of_inputs.append({"n": 15, "dtype": np.float16})

    # Input 10: 100x100 int16 identity matrix
    list_of_inputs.append({"n": 100, "dtype": np.int16})

    return list_of_inputs

generated_inputs["jax.numpy.identity"] = identity_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.identity'.")


check_valid('jax.numpy.identity', generated_inputs['jax.numpy.identity'], lib="jax", suffix=0)
