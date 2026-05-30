
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctan2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    x2 = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 2D float32 arrays
    x1 = np.random.randn(3, 3).astype(np.float32)
    x2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: float64 arrays with negative and positive values
    x1 = np.array([[-2.5, 3.5], [-0.5, 0.0]], dtype=np.float64)
    x2 = np.array([[1.5, -2.5], [0.0, -1.0]], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting (1D and 2D arrays)
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Broadcasting (2D column and row vectors)
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: 3D float32 arrays
    x1 = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: 0D scalar-like arrays
    x1 = np.array(1.5, dtype=np.float32)
    x2 = np.array(-1.5, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: Integer arrays (should be cast to float)
    x1 = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    x2 = np.array([2, 1, 0, -1, -2], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Large dimensions
    x1 = np.random.randn(64, 64).astype(np.float32)
    x2 = np.random.randn(64, 64).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: 4D arrays
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 11: Mixed int64 inputs
    x1 = np.arange(-5, 5, dtype=np.int64)
    x2 = np.arange(5, -5, -1, dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.arctan2"] = arctan2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan2'.")


check_valid('jax.numpy.arctan2', generated_inputs['jax.numpy.arctan2'], lib="jax", suffix=0)
