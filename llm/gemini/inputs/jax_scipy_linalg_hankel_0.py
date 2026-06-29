
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hankel_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer arrays (positive values)
    c = np.array([1, 2, 3], dtype=np.int32)
    r = np.array([3, 4, 5, 6], dtype=np.int32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 1D float arrays with negative values
    c = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    r = np.array([-3.0, -4.0, -5.0], dtype=np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 arrays with different lengths
    c = np.array([10.5, 20.5], dtype=np.float64)
    r = np.array([20.5, 30.5, 40.5, 50.5], dtype=np.float64)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched 2D inputs of same shape
    c = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    r = np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched input with broadcasting (1D c and 2D r)
    c = np.array([1, 2, 3], dtype=np.float32)
    r = np.array([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched input with broadcasting (2D c and 1D r)
    c = np.array([[1, 2], [3, 4]], dtype=np.int32)
    r = np.array([2, 5, 6], dtype=np.int32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D batched inputs with floats
    c = np.random.randn(2, 2, 3).astype(np.float32)
    r = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex number arrays
    c = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    r = np.array([3+3j, 4+4j, 5+5j], dtype=np.complex64)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element arrays
    c = np.array([5.0], dtype=np.float32)
    r = np.array([5.0, 6.0], dtype=np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean arrays
    c = np.array([True, False, True], dtype=bool)
    r = np.array([True, True, False, False], dtype=bool)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.hankel"] = hankel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.hankel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.hankel'.")


check_valid('jax.scipy.linalg.hankel', generated_inputs['jax.scipy.linalg.hankel'], lib="jax", suffix=0)
