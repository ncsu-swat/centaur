
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def geom_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, matching shapes
    k = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 2: Scalars (0D arrays)
    k = np.array(3, dtype=np.int32)
    p = np.array(0.3, dtype=np.float32)
    loc = np.array(1, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 3: float64, 1D arrays
    k = np.array([2, 4, 6], dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([0, 0, 0], dtype=np.float64)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 4: 2D arrays, matching shapes
    k = np.array([[1, 2], [3, 4]], dtype=np.int32)
    p = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[0, 0], [1, 1]], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 5: Broadcasting, k is 2D, p is 1D, loc is 0D
    k = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    p = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 6: Large k values, float32
    k = np.array([10, 20, 30], dtype=np.int32)
    p = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    loc = np.array([5, 5, 5], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 7: Values of k below loc (should yield 0)
    k = np.array([1, 2, 3], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([5, 5, 5], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 8: Floating point k values
    k = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 9: 3D arrays
    k = np.ones((2, 2, 2), dtype=np.int32) * 2
    p = np.ones((2, 2, 2), dtype=np.float32) * 0.25
    loc = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 10: p close to boundaries
    k = np.array([1, 2], dtype=np.int32)
    p = np.array([1e-5, 0.9999], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    return list_of_inputs

generated_inputs["jax.scipy.stats.geom.pmf"] = geom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.geom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.geom.pmf'.")


check_valid('jax.scipy.stats.geom.pmf', generated_inputs['jax.scipy.stats.geom.pmf'], lib="jax", suffix=0)
