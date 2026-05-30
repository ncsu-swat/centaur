
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tri_inputs():
    list_of_inputs = []

    # Input 1: Square matrix with main diagonal
    input_dict = {
        "N": 3,
        "M": 3,
        "k": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix with negative k
    input_dict = {
        "N": 4,
        "M": 5,
        "k": -1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix with positive k and integer dtype
    input_dict = {
        "N": 5,
        "M": 3,
        "k": 2,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Minimal size with boolean dtype
    input_dict = {
        "N": 1,
        "M": 1,
        "k": 0,
        "dtype": np.bool_
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger square matrix with float64 dtype and deep negative k
    input_dict = {
        "N": 10,
        "M": 10,
        "k": -5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large rectangular matrix with int64 dtype and positive k
    input_dict = {
        "N": 8,
        "M": 12,
        "k": 4,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Medium square matrix with uint32 dtype and negative k
    input_dict = {
        "N": 6,
        "M": 6,
        "k": -3,
        "dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tall rectangular matrix with complex64 dtype
    input_dict = {
        "N": 15,
        "M": 10,
        "k": 0,
        "dtype": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small rectangular matrix with float16 dtype and out-of-bounds negative k
    input_dict = {
        "N": 2,
        "M": 4,
        "k": -2,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Medium square matrix with int16 dtype and out-of-bounds positive k
    input_dict = {
        "N": 7,
        "M": 7,
        "k": 10,
        "dtype": np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tri"] = tri_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tri'.")


check_valid('jax.numpy.tri', generated_inputs['jax.numpy.tri'], lib="jax", suffix=0)
