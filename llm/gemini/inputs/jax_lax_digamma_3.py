
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32 representing integer 5
    input_dict = {"x": np.float32(5.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float64 representing integer 10
    input_dict = {"x": np.float64(10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of float32 representing integers
    input_dict = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of float64 representing integers
    input_dict = {"x": np.array([10.0, 20.0, 30.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of float32 representing integers
    input_dict = {"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of float64 representing integers
    input_dict = {"x": np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array of float32 representing integers
    input_dict = {"x": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of float32 representing a single integer
    input_dict = {"x": np.array([42.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array of float32 representing integers
    input_dict = {"x": (np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of float32 representing integers
    input_dict = {"x": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.digamma_3"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_3'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_3'], lib="jax", suffix=3)
