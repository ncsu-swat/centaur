
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_add_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D integer arrays with negative values
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Scalar arrays (0-D tensors)
    x = np.array(10.5, dtype=np.float64)
    y = np.array(-2.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting 1D array to 2D array
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting 3D array with 1D array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(1).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Large 2D float64 arrays
    x = np.random.randn(100, 100).astype(np.float64)
    y = np.random.randn(100, 100).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 4D tensors with positive/negative float32
    x = np.random.uniform(-10, 10, size=(2, 2, 3, 3)).astype(np.float32)
    y = np.random.uniform(-10, 10, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Complex number arrays
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    y = np.array([5 - 6j, 7 + 8j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Int16 arrays to test smaller integer types
    x = np.array([[10, 20], [30, 40]], dtype=np.int16)
    y = np.array([[1, 2], [3, 4]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Broadcasting with mismatched singleton dimensions (e.g., (3, 1) and (1, 4))
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(1, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.add_1"] = jax_numpy_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_1'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_1'], lib="jax", suffix=1)
