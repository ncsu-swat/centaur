
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def toeplitz_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays of float32
    c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    r = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard 1D arrays of int32, different sizes
    c = np.array([1, 2, 3], dtype=np.int32)
    r = np.array([4, 5, 6, 7], dtype=np.int32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 1D arrays
    c = np.array([1+1j, 2+2j], dtype=np.complex64)
    r = np.array([3+3j, 4+4j, 5+5j], dtype=np.complex64)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D batched float32 with identical batch size
    c = np.random.randn(2, 3).astype(np.float32)
    r = np.random.randn(2, 4).astype(np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D batched float32 with broadcasting batch size
    c = np.random.randn(3, 4).astype(np.float32)
    r = np.random.randn(1, 5).astype(np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 with negative values
    c = np.array([-1.5, -2.5, -3.5], dtype=np.float64)
    r = np.array([-1.5, -4.5], dtype=np.float64)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D batched float32
    c = np.random.randn(2, 2, 3).astype(np.float32)
    r = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D with single element in column
    c = np.array([5.0], dtype=np.float32)
    r = np.array([5.0, 6.0], dtype=np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D batched int64
    c = np.random.randint(-10, 10, (2, 5)).astype(np.int64)
    r = np.random.randint(-10, 10, (2, 5)).astype(np.int64)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex128
    c = np.array([1-1j, 2-2j, 3-3j], dtype=np.complex128)
    r = np.array([1-1j, 5-5j, 6-6j], dtype=np.complex128)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Broadcasting with c batch dimension equal to 1
    c = np.random.randn(1, 3).astype(np.float32)
    r = np.random.randn(4, 3).astype(np.float32)
    input_dict = {"c": c, "r": r}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.toeplitz"] = toeplitz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.toeplitz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.toeplitz'.")


check_valid('jax.scipy.linalg.toeplitz', generated_inputs['jax.scipy.linalg.toeplitz'], lib="jax", suffix=0)
