
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccosh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values >= 1
    x = np.array([1.0, 1.5, 2.0, 3.0, 10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of float32 values >= 1
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of int32 values >= 1
    x = np.array([1, 2, 5, 10, 50], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array including values < 1 and negative values (produces NaN)
    x = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 2.5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of float64 values
    x = np.arange(1.0, 9.0, dtype=np.float64).reshape((2, 2, 2))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of complex64 values
    x = np.array([1.0 + 2.0j, -5.0 + 0.0j, 0.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of complex128 values
    x = np.array(
        [[1.0 + 1.0j, 2.0 - 3.0j], [-1.0 + 0.0j, 0.5 + 0.5j]],
        dtype=np.complex128,
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with single element
    x = np.array([2.5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with special values (inf and nan)
    x = np.array([1.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of float32 values
    x = np.linspace(1.0, 10.0, 16, dtype=np.float32).reshape((2, 2, 2, 2))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int64 array
    x = np.array([1, 3, 7, 15, 31], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arccosh_5"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_5'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_5'], lib="jax", suffix=5)
