
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def poly_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    seq_of_zeros = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 2: 1D integer array
    seq_of_zeros = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 3: 1D float array with negative values and zero
    seq_of_zeros = np.array([-2.5, 0.0, 3.5, -1.0], dtype=np.float32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 4: 1D complex array (complex conjugates)
    seq_of_zeros = np.array([2.0 + 0.0j, 1.0 + 2.0j, 1.0 - 2.0j], dtype=np.complex64)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 5: 1D float64 array
    seq_of_zeros = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 6: 1D single-element array
    seq_of_zeros = np.array([-7.0], dtype=np.float32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 7: 2D square matrix (2x2 float32)
    seq_of_zeros = np.array([[2.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 8: 2D square matrix (3x3 float64)
    seq_of_zeros = np.array(
        [[2.0, 1.0, 5.0], [3.0, 4.0, 7.0], [1.0, 3.0, 5.0]], dtype=np.float64
    )
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 9: 2D square matrix with complex values (2x2 complex128)
    seq_of_zeros = np.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [0.0 + 3.0j, 4.0 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 10: 2D 1x1 matrix (float32)
    seq_of_zeros = np.array([[3.14]], dtype=np.float32)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    # Input 11: 1D array of int64 with negative and positive values
    seq_of_zeros = np.array([-5, -3, 0, 3, 5], dtype=np.int64)
    list_of_inputs.append({"seq_of_zeros": copy.deepcopy(seq_of_zeros)})

    return list_of_inputs


generated_inputs["jax.numpy.poly_1"] = poly_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.poly_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.poly_1'.")


check_valid('jax.numpy.poly', generated_inputs['jax.numpy.poly_1'], lib="jax", suffix=1)
