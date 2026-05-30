
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def copysign_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays of the same shape (float32)
    x1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x2 = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: Broadcasting 2D and 1D arrays
    x1 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    x2 = np.array([-1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Float64 3D arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Integer arrays (np.int32)
    x1 = np.array([5, -10, 15], dtype=np.int32)
    x2 = np.array([-1, -2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Special values (zeros with different signs)
    x1 = np.array([0.0, -0.0, 0.0], dtype=np.float32)
    x2 = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Special values (Infs and NaNs)
    x1 = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    x2 = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Broadcasting with a single element array
    x1 = np.random.randn(3, 3).astype(np.float32)
    x2 = np.array([-1.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Larger dimensions (4D)
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Mixed signs and zero in x2 (using float values to preserve -0.0 sign)
    x1 = np.array([-5.5, 4.4, -3.3], dtype=np.float32)
    x2 = np.array([-0.0, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Int64 arrays
    x1 = np.array([[10, -20], [-30, 40]], dtype=np.int64)
    x2 = np.array([[-1, 1], [1, -1]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: 1D broadcasting with higher dimensional x1
    x1 = np.random.randn(1, 5).astype(np.float32)
    x2 = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.copysign"] = copysign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.copysign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.copysign'.")


check_valid('jax.numpy.copysign', generated_inputs['jax.numpy.copysign'], lib="jax", suffix=0)
