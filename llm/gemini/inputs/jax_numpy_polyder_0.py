
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyder_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D array, order 1
    p = np.array([2.0, -5.0, 3.0, -1.0], dtype=np.float32)
    m = 1
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 2: Int32 1D array, order 2
    p = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    m = 2
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 3: Float64 1D array, order 1
    p = np.array([0.5, -2.5, 10.0], dtype=np.float64)
    m = 1
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 4: Negative and positive values, order 3
    p = np.array([-10, 20, -30, 40, -50], dtype=np.int64)
    m = 3
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 5: Single element polynomial, order 1 (derivative should be 0)
    p = np.array([42.0], dtype=np.float32)
    m = 1
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 6: Higher order derivative on large polynomial
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    m = 4
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 7: All zeros polynomial, order 2
    p = np.zeros(10, dtype=np.float32)
    m = 2
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 8: Very large order of differentiation (exceeding degree, returns empty/zero)
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = 5
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 9: Large random values, float64, order 1
    p = np.random.uniform(-100, 100, size=20).astype(np.float64)
    m = 1
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    # Input 10: Array with small decimals, order 2
    p = np.array([1e-3, -2e-4, 5e-5, -1e-6], dtype=np.float32)
    m = 2
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m})

    return list_of_inputs

generated_inputs["jax.numpy.polyder"] = polyder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyder'.")


check_valid('jax.numpy.polyder', generated_inputs['jax.numpy.polyder'], lib="jax", suffix=0)
