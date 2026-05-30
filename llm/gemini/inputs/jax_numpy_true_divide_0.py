
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def true_divide_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 arrays
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 2D float64 arrays with negative values
    x1 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, -2.0], [2.0, -2.0]], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: 3D float32 and 1D float32 (Broadcasting)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Integer arrays (int32)
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = np.array([3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Complex arrays (complex64)
    x1 = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    x2 = np.array([2j, 1 - 1j], dtype=np.complex64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: 1-element 0D arrays (scalars represented as arrays)
    x1 = np.array(5.0, dtype=np.float32)
    x2 = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: 4D float32 arrays
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: Mixed types - x1 is float32, x2 is int32
    x1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    x2 = np.array([2, 2, 2], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Large values, float64
    x1 = np.array([1e10, -2e10], dtype=np.float64)
    x2 = np.array([3e5, 4e5], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: Broadcasting 1D and 2D arrays
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 11: Unsigned integers (uint8)
    x1 = np.array([10, 20, 30], dtype=np.uint8)
    x2 = np.array([3, 3, 3], dtype=np.uint8)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.true_divide"] = true_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.true_divide'.")


check_valid('jax.numpy.true_divide', generated_inputs['jax.numpy.true_divide'], lib="jax", suffix=0)
