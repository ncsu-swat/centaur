
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_frexp_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array with positive numbers
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative numbers
    x = np.array([-1.0, -2.5, -0.75, -10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    x = np.random.uniform(-100.0, 100.0, size=(3, 3)).astype(np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar (0D array) float32
    x = np.array(3.14, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float16 array
    x = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array with large values
    x = np.array([[[[1e5, 2e10], [3e15, 4e20]]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array with very small values
    x = np.array([[1e-10, 2e-20], [3e-30, 4e-40]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array containing zeros
    x = np.array([0.0, -0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive, negative, and zero values in 1D array
    x = np.array([-100.0, 0.0, 100.0, -0.001, 0.001], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.frexp"] = jax_numpy_frexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.frexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.frexp'.")


check_valid('jax.numpy.frexp', generated_inputs['jax.numpy.frexp'], lib="jax", suffix=0)
