
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyder_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array, m=1
    p = np.array([2.0, -5.0, 3.0, -1.0], dtype=np.float32)
    m = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 2: 1D float64 array, second order derivative (m=2)
    p = np.array([3.0, 0.0, -2.0, 4.0, -7.0], dtype=np.float64)
    m = 2
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 3: 1D int32 array with negative coefficients, m=1
    p = np.array([-4, 7, -2, 9], dtype=np.int32)
    m = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 4: 1D int64 array, higher order differentiation (m=3)
    p = np.array([1, -3, 3, -1, 5, -6], dtype=np.int64)
    m = 3
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 5: Complex numbers (complex64), m=1
    p = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    m = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 6: Complex numbers (complex128), m=2
    p = np.array(
        [2.5 - 1.5j, 0.0 + 3.0j, -1.0 - 2.0j, 5.0 + 0.0j], dtype=np.complex128
    )
    m = 2
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 7: Constant polynomial (length 1), m=1
    p = np.array([42.0], dtype=np.float32)
    m = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 8: Polynomial degree smaller than order of differentiation, m=4
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    m = 4
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 9: High degree polynomial, m=5
    p = np.linspace(-10.0, 10.0, num=12, dtype=np.float64)
    m = 5
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 10: All zero polynomial coefficients, m=1
    p = np.zeros(6, dtype=np.float32)
    m = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

    # Input 11: Order of differentiation m=0 (identity)
    p = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    m = 0
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m}))

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
