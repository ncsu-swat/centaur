
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def poly_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive values
    seq = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, float64, mixed positive/negative values
    seq = np.array([-1.5, 0.0, 2.5, -3.0, 4.2], dtype=np.float64)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array (square matrix), float32
    seq = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array (square matrix), float64, mixed values
    seq = np.array([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0], [0.0, 1.0, -2.0]], dtype=np.float64)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, complex64, with complex roots
    seq = np.array([1.0 + 2.0j, 1.0 - 2.0j, -3.0j], dtype=np.complex64)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, int32 values
    seq = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, int64 values with negatives
    seq = np.array([-5, -10, 15], dtype=np.int64)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array (square matrix), complex128
    seq = np.array([[1.0 + 1.0j, 2.0j], [-2.0j, 1.0 - 1.0j]], dtype=np.complex128)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, float32, length 1
    seq = np.array([5.5], dtype=np.float32)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array (square matrix), float32, shape (4, 4)
    seq = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"seq_of_zeros": seq}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
