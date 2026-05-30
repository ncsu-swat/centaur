
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pow_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays with positive bases
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([2.0, 0.5, -1.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 2D float32 base with 0D scalar-like exponent
    x1 = np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float32)
    x2 = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: 2D int32 arrays with non-negative exponents
    x1 = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    x2 = np.random.randint(0, 5, size=(4, 4)).astype(np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Negative float bases with integer-valued float exponents
    x1 = np.array([-1.5, -2.0, -3.5], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: float64 with broadcasting dimensions
    x1 = np.random.uniform(0.1, 10.0, size=(1, 5)).astype(np.float64)
    x2 = np.random.uniform(-2.0, 2.0, size=(5, 1)).astype(np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: complex64 arrays
    x1 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    x2 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: 3D float32 arrays
    x1 = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: int64 arrays
    x1 = np.random.randint(1, 10, size=(5,)).astype(np.int64)
    x2 = np.random.randint(0, 3, size=(5,)).astype(np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: float32 base with exponent of all zeros
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.zeros(5, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: 4D float32 arrays
    x1 = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(1.0, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.pow"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pow'.")


check_valid('jax.numpy.pow', generated_inputs['jax.numpy.pow'], lib="jax", suffix=0)
