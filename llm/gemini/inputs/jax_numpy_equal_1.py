
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays of the same shape
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D int32 arrays of the same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 5], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Broadcasting 1D array with 2D array
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([1, 5, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Scalar-like 0-D arrays
    x = np.array(5.0, dtype=np.float64)
    y = np.array(5.0, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Complex numbers
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    y = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Boolean arrays
    x = np.array([True, False, True], dtype=bool)
    y = np.array([False, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Arrays with NaNs and infinities
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    y = np.array([np.nan, np.inf, np.inf, -0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: High-dimensional float64 arrays (3D)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = copy.deepcopy(x)
    y[0, 0, 0] = 0.0  # Introduce one mismatch
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Advanced broadcasting (2, 1, 3) and (1, 4, 3)
    x = np.random.randint(0, 5, size=(2, 1, 3)).astype(np.int32)
    y = np.random.randint(0, 5, size=(1, 4, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Negative values and larger integers (int64)
    x = np.array([-100, -200, 300], dtype=np.int64)
    y = np.array([-100, 200, 300], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Empty arrays
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.equal_1"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_1'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_1'], lib="jax", suffix=1)
