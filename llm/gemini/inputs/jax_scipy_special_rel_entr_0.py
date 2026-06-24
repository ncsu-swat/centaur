
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rel_entr_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 positive arrays
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    q = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 2: 2D float32 positive arrays
    p = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    q = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 3: Float32 with zeros in p (valid edge case resulting in 0)
    p = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 4: Large dimensions with float64 positive arrays
    p = np.random.uniform(0.1, 10.0, size=(3, 4, 5)).astype(np.float64)
    q = np.random.uniform(0.1, 10.0, size=(3, 4, 5)).astype(np.float64)
    list_of_inputs.append({"p": p, "q": q})

    # Input 5: 0-D arrays (scalar equivalents)
    p = np.array(1.5, dtype=np.float32)
    q = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 6: Negative values (tests "otherwise" condition resulting in infinity)
    p = np.array([-1.0, 2.0, 3.0], dtype=np.float32)
    q = np.array([2.0, -2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 7: Broadcasting behavior with row and column vectors
    p = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    q = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 8: 4D arrays with float32 positive arrays
    p = np.random.exponential(scale=1.0, size=(2, 2, 3, 3)).astype(np.float32) + 0.01
    q = np.random.exponential(scale=1.0, size=(2, 2, 3, 3)).astype(np.float32) + 0.01
    list_of_inputs.append({"p": p, "q": q})

    # Input 9: Very small positive float32 values
    p = np.array([1e-5, 2e-5, 3e-5], dtype=np.float32)
    q = np.array([1e-4, 1e-4, 1e-4], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    # Input 10: Large positive float64 values
    p = np.array([1e5, 2e5, 3e5], dtype=np.float64)
    q = np.array([1e4, 1e4, 1e4], dtype=np.float64)
    list_of_inputs.append({"p": p, "q": q})

    # Input 11: Mixed case with zeros, negatives, and positive values
    p = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    q = np.array([0.5, 0.5, -0.5, 1.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "q": q})

    return list_of_inputs

generated_inputs["jax.scipy.special.rel_entr"] = rel_entr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.rel_entr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.rel_entr'.")


check_valid('jax.scipy.special.rel_entr', generated_inputs['jax.scipy.special.rel_entr'], lib="jax", suffix=0)
