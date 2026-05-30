
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cosh_inputs():
    list_of_inputs = []

    # Input 1: float32 array
    input_dict = {"x": np.array([0.0, 1.0, -1.0, 2.5, -2.5], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 array
    input_dict = {"x": np.array([0, 1, -1, 2, -2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    input_dict = {"x": np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array
    input_dict = {"x": np.array([[[0.5, -0.5], [1.5, -1.5]], [[2.5, -2.5], [3.5, -3.5]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 array
    input_dict = {"x": np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.0 + 0.0j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with single element
    input_dict = {"x": np.array([0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 array with small and moderate values
    input_dict = {"x": np.array([1e-5, -1e-5, 20.0, -20.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 array
    input_dict = {"x": np.array([1.0, 2.5, -3.0, 4.2], dtype=np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array
    input_dict = {"x": np.array([[[[1.0, -1.0]]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with wider range
    input_dict = {"x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cosh_4"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_4'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_4'], lib="jax", suffix=4)
