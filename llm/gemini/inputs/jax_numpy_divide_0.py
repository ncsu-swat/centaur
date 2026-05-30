
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divide_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 arrays
    x1 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    x2 = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D float32 arrays (matrix shape)
    x1 = np.random.randn(3, 3).astype(np.float32)
    x2 = np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 3D float32 arrays with negative values
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(1.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Int32 arrays
    x1 = np.array([10, -20, 30], dtype=np.int32)
    x2 = np.array([3, 4, -5], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Large dimension arrays (4D tensor)
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Float64 arrays
    x1 = np.random.randn(5).astype(np.float64)
    x2 = np.random.uniform(1.0, 5.0, size=(5,)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Broadcasting (3, 1) and (1, 3) arrays
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 4.0, 8.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Broadcasting a scalar-like 1D array to a 2D array
    x1 = np.random.randn(2, 3).astype(np.float32)
    x2 = np.array([2.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Complex64 arrays
    x1 = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    x2 = np.array([2 + 1j, 1 - 1j], dtype=np.complex64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Boolean arrays
    x1 = np.array([True, False, True], dtype=np.bool_)
    x2 = np.array([True, True, True], dtype=np.bool_)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.divide"] = divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divide'.")


check_valid('jax.numpy.divide', generated_inputs['jax.numpy.divide'], lib="jax", suffix=0)
