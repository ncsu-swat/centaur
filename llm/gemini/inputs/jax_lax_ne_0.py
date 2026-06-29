
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_ne_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, identical shapes
    x = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    y = np.array([1.0, 2.5, -3.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D int32 arrays with negative values, identical shapes
    x = np.array([[-1, 2, 3], [4, -5, 6]], dtype=np.int32)
    y = np.array([[1, 2, -3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: float64 arrays, broadcasting on second dimension (same number of dimensions)
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Boolean arrays
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([[False, False], [True, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 3D float32 arrays, identical shapes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: int8 arrays, broadcasting on first dimension (same number of dimensions)
    x = np.array([[1, 2, 3]], dtype=np.int8)
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Complex64 arrays, identical shapes
    x = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    y = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 0D scalar-like arrays (0-dimensional tensors)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: uint8 arrays, identical shapes
    x = np.array([0, 255, 128], dtype=np.uint8)
    y = np.array([0, 254, 128], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 4D float32 arrays, broadcasting multiple dimensions
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Large int64 arrays, identical shapes
    x = np.arange(100, dtype=np.int64)
    y = np.arange(100, dtype=np.int64)
    y[50] = 999  # introduce one discrepancy
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.ne"] = jax_lax_ne_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ne' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ne'.")


check_valid('jax.lax.ne', generated_inputs['jax.lax.ne'], lib="jax", suffix=0)
