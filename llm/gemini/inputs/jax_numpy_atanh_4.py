
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    input_dict = {"x": np.array([0.0, 0.5, -0.5, 0.25, -0.75], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of float64
    input_dict = {"x": np.array([-0.9, -0.4, 0.0, 0.4, 0.9], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single element 1D array
    input_dict = {"x": np.array([0.3], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with values close to domain limits
    input_dict = {
        "x": np.array([-0.999, -0.5, 0.0, 0.5, 0.999], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of complex64 numbers
    input_dict = {
        "x": np.array([0.2 + 0.3j, -0.5 + 0.1j, 0.0 - 0.4j], dtype=np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of complex128 numbers
    input_dict = {
        "x": np.array([0.5 + 0.5j, -0.8 - 0.2j, 0.1 + 0.9j], dtype=np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with two elements
    input_dict = {"x": np.array([-0.2, 0.2], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with small negative values
    input_dict = {
        "x": np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with symmetrically spaced values
    input_dict = {
        "x": np.array(
            [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with tiny values around zero
    input_dict = {
        "x": np.array([-1e-5, -1e-7, 0.0, 1e-7, 1e-5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array of single complex element
    input_dict = {"x": np.array([0.0 + 0.0j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.atanh_4"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_4'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_4'], lib="jax", suffix=4)
