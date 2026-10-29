
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    n = np.array(0, dtype=np.int32)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 2: 1D arrays of matching shape
    n = np.array([0, 1, 2, 3], dtype=np.int32)
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 3: 2D arrays, float64
    n = np.array([[1, 2], [3, 4]], dtype=np.int64)
    x = np.array([[0.5, 1.2], [2.3, 4.1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 4: Broadcasting scalar n with 2D x
    n = np.array(1, dtype=np.int32)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 5: Broadcasting column n with row x
    n = np.array([[0], [1], [2]], dtype=np.int32)
    x = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 6: Negative non-integer values of x
    n = np.array([0, 1, 0], dtype=np.int32)
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 7: 3D arrays
    n = np.zeros((2, 3, 2), dtype=np.int32)
    x = np.ones((2, 3, 2), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 8: Higher order derivatives with large x
    n = np.array([5, 8], dtype=np.int64)
    x = np.array([50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 9: Small fractional x values between 0 and 1
    n = np.array([1, 2, 3], dtype=np.int32)
    x = np.array([0.01, 0.1, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 10: 4D broadcastable arrays
    n = np.array([[[[1]]]], dtype=np.int32)
    x = np.array([[[[1.2, 2.3], [3.4, 4.5]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 11: Digamma order (n = 0) with 1D array of mixed floats
    n = np.array([0, 0, 0, 0], dtype=np.int32)
    x = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.polygamma_2"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_2'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_2'], lib="jax", suffix=2)
