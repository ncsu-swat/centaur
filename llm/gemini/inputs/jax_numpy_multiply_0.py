
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def multiply_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays of same size
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D float32 arrays of same shape with negative values
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, -6.0], [7.0, -8.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Broadcasting 1D array to 2D array
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting with compatible shapes (1, 3) and (3, 1)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: High dimensional arrays (4D and 4D)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Integer arrays (int32)
    x = np.array([1, -2, 3, -4], dtype=np.int32)
    y = np.array([10, 10, 10, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Float64 arrays for precision
    x = np.random.randn(5, 5).astype(np.float64)
    y = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Boolean arrays (logical element-wise multiplication)
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Scalar-like 0D arrays (represented as 0D numpy arrays)
    x = np.array(5.5, dtype=np.float32)
    y = np.array(-2.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Complex numbers multiplication
    x = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    y = np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Large dimensions with broadcasting
    x = np.random.randn(1, 100, 100).astype(np.float32)
    y = np.random.randn(5, 1, 100).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.multiply"] = multiply_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.multiply'.")


check_valid('jax.numpy.multiply', generated_inputs['jax.numpy.multiply'], lib="jax", suffix=0)
