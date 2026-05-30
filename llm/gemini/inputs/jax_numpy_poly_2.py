
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def poly_inputs():
    list_of_inputs = []

    # Input 1: Simple positive scalar integer
    seq_of_zeros = 5
    list_of_inputs.append({"seq_of_zeros": seq_of_zeros})

    # Input 2: Negative scalar integer
    seq_of_zeros = -3
    list_of_inputs.append({"seq_of_zeros": seq_of_zeros})

    # Input 3: Zero scalar integer
    seq_of_zeros = 0
    list_of_inputs.append({"seq_of_zeros": seq_of_zeros})

    # Input 4: 1D numpy array of positive integers (int32)
    seq_of_zeros = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 5: 1D numpy array of negative integers (int64)
    seq_of_zeros = np.array([-5, -10, -15], dtype=np.int64)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 6: 1D numpy array with mixed positive, negative, and zero integers
    seq_of_zeros = np.array([-2, 0, 4, -1, 3], dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 7: 2D square matrix of integers (shape 2x2)
    seq_of_zeros = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 8: 2D square matrix of integers with negative values (shape 3x3)
    seq_of_zeros = np.array([[2, -1, 0], [3, 4, -2], [1, 0, 5]], dtype=np.int64)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 9: Large 1D array of integers
    seq_of_zeros = np.arange(-10, 11, 2, dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 10: Single-element 1D array of integers
    seq_of_zeros = np.array([42], dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 11: A large scalar integer
    seq_of_zeros = 1000
    list_of_inputs.append({"seq_of_zeros": seq_of_zeros})

    return list_of_inputs

generated_inputs["jax.numpy.poly_2"] = poly_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.poly_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.poly_2'.")


check_valid('jax.numpy.poly', generated_inputs['jax.numpy.poly_2'], lib="jax", suffix=2)
