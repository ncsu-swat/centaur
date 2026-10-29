
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyint_inputs():
    list_of_inputs = []

    # Input 1: Basic 1st order integration with float32
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = np.array([4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 2: 2nd order integration with float64
    p = np.array([12.0, 12.0, 6.0], dtype=np.float64)
    m = 2
    k = np.array([4.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 3: Negative and zero coefficients with int32
    p = np.array([-3, 0, 5], dtype=np.int32)
    m = 1
    k = np.array([2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 4: 3rd order integration with float32 constants
    p = np.array([2.5, -1.5, 0.0, 4.2], dtype=np.float32)
    m = 3
    k = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 5: Single coefficient polynomial (constant polynomial)
    p = np.array([5.0], dtype=np.float64)
    m = 1
    k = np.array([-2.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 6: Complex numbers polynomial
    p = np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64)
    m = 2
    k = np.array([0.0 + 1.0j, -1.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 7: Higher order integration (m=4) with int64
    p = np.array([1, -1, 1, -1], dtype=np.int64)
    m = 4
    k = np.array([0, 1, -2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 8: Zero integration constant
    p = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    m = 1
    k = np.array([0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 9: Order m=0 (identity integration) with empty constant array
    p = np.array([4.0, -2.0, 7.0], dtype=np.float64)
    m = 0
    k = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 10: High degree polynomial with 2nd order integration
    p = np.array([1.0, 0.0, -2.0, 0.0, 3.0, -4.0], dtype=np.float32)
    m = 2
    k = np.array([10.5, -5.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.polyint_2"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_2'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_2'], lib="jax", suffix=2)
