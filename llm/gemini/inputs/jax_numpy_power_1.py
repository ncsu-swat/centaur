
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 1D int32 arrays with positive exponents
    x1 = np.array([2, 3, 4, 5], dtype=np.int32)
    x2 = np.array([3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: 2D float64 arrays
    x1 = np.random.uniform(1.0, 5.0, (3, 3)).astype(np.float64)
    x2 = np.random.uniform(1.0, 3.0, (3, 3)).astype(np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting (1D base and 2D exponent)
    x1 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: 0-D arrays (representing scalars as tensors)
    x1 = np.array(5.0, dtype=np.float32)
    x2 = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: Negative base values with integer exponents (to avoid nan/complex issues)
    x1 = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    x2 = np.array([3.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: 3D float32 arrays
    x1 = np.random.uniform(1.0, 2.0, (2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(-1.0, 1.0, (2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: 1D int64 arrays
    x1 = np.array([2, 5, 10], dtype=np.int64)
    x2 = np.array([5, 3, 2], dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Complex64 arrays
    x1 = np.array([1 + 1j, 2 - 3j], dtype=np.complex64)
    x2 = np.array([2.0, 3.0], dtype=np.complex64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: Highly broadcasted shapes (1, 5, 1) and (4, 1, 3)
    x1 = np.ones((1, 5, 1), dtype=np.float32) * 2.0
    x2 = np.ones((4, 1, 3), dtype=np.float32) * 3.0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.power_1"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_1'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_1'], lib="jax", suffix=1)
