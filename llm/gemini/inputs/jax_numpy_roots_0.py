
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def roots_inputs():
    list_of_inputs = []

    # Input 1: Quadratic polynomial, float32, strip_zeros=True
    p = np.array([1.0, -3.0, 2.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 2: Linear polynomial, float64, strip_zeros=False
    p = np.array([2.0, 4.0], dtype=np.float64)
    strip_zeros = False
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 3: Cubic polynomial with negative values, float32, strip_zeros=True
    p = np.array([-1.0, 0.0, 4.0, -5.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 4: Polynomial with leading zeros, float32, strip_zeros=True
    p = np.array([0.0, 0.0, 1.0, -2.0, 1.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 5: Polynomial with leading zeros, float64, strip_zeros=False
    p = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    strip_zeros = False
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 6: Complex coefficients, complex64, strip_zeros=True
    p = np.array([1.0 + 2.0j, -3.0 + 0.0j, 1.0 - 1.0j], dtype=np.complex64)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 7: Complex coefficients, complex128, strip_zeros=False
    p = np.array([0.0 + 0.0j, 2.0 - 1.0j, 4.0 + 3.0j], dtype=np.complex128)
    strip_zeros = False
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 8: Integer coefficients, int32, strip_zeros=True
    p = np.array([1, -5, 6], dtype=np.int32)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 9: Integer coefficients, int64, strip_zeros=False
    p = np.array([3, -2, -1], dtype=np.int64)
    strip_zeros = False
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 10: Higher degree polynomial (quartic), float64, strip_zeros=True
    p = np.array([1.0, 0.0, -10.0, 0.0, 9.0], dtype=np.float64)
    strip_zeros = True
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    # Input 11: Higher degree polynomial (degree 6), float32, strip_zeros=False
    p = np.array([2.5, -1.2, 0.4, -3.1, 1.0, -0.5, 2.0], dtype=np.float32)
    strip_zeros = False
    list_of_inputs.append(
        copy.deepcopy({"p": p, "strip_zeros": strip_zeros})
    )

    return list_of_inputs


generated_inputs["jax.numpy.roots"] = roots_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roots' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roots'.")


check_valid('jax.numpy.roots', generated_inputs['jax.numpy.roots'], lib="jax", suffix=0)
