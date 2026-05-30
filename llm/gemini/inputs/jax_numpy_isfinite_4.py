
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    input_dict = {"x": np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with inf and nan values
    input_dict = {"x": np.array([1.0, np.inf, -np.inf, np.nan], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of integers
    input_dict = {"x": np.array([-100, -1, 0, 1, 100], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of float64 with inf and nan
    input_dict = {"x": np.array([[1.0, 2.0, np.inf], [-1.0, np.nan, 0.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of float32
    input_dict = {"x": np.array([[[1.0, 2.0], [3.0, 4.0]], [[np.inf, -np.inf], [0.0, np.nan]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of complex numbers
    input_dict = {"x": np.array([3 - 4j, 1 + 2j, np.inf + 0j, np.nan * 1j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of booleans
    input_dict = {"x": np.array([True, False, True, False], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large 1D array of random float32
    input_dict = {"x": np.random.randn(50).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extremely small and large finite floats
    input_dict = {"x": np.array([1e-30, -1e-30, 1e30, -1e30], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of complex128
    input_dict = {"x": np.array([[1j, np.inf * 1j], [np.nan + 2j, 0j]], dtype=np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_4"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_4'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_4'], lib="jax", suffix=4)
