
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: 1D array of complexes
    input_dict = {"x": np.array([1+1j, 2-3j, -4j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of integers
    input_dict = {"x": np.array([1, -2, 3, 0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of floats
    input_dict = {"x": np.array([1.5, -2.5, 0.0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of complexes
    input_dict = {"x": np.array([[1+2j, 3-4j], [5+6j, 7-8j]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of mixed reals and complexes
    input_dict = {"x": np.array([[1.0, 2-3j], [3j, -4.5]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of complexes
    input_dict = {"x": np.array([[[1j, 2j], [3j, 4j]], [[5j, 6j], [7j, 8j]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of booleans
    input_dict = {"x": np.array([True, False, True])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array containing complex numbers with NaN and Inf
    input_dict = {"x": np.array([complex(np.nan, np.inf), complex(1.0, -np.nan)])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element array with complex zero
    input_dict = {"x": np.array([0.0+0.0j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D array of complex numbers
    large_arr = np.array([complex(float(i), float(-i)) for i in range(50)])
    input_dict = {"x": large_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_4"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_4'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_4'], lib="jax", suffix=4)
