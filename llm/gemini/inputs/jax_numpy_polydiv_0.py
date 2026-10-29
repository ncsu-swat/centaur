
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polydiv_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D polynomials, len(u) > len(v), trim_leading_zeros=False
    u = np.array([3.0, 5.0, 2.0], dtype=np.float32)
    v = np.array([2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 2: Float32 polynomials, trim_leading_zeros=True
    u = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    v = np.array([4.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": True})
    )

    # Input 3: Integer arrays (int32)
    u = np.array([1, -3, 3, -1], dtype=np.int32)
    v = np.array([1, -1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 4: len(u) < len(v), quotient will be zero
    u = np.array([2.0, 3.0], dtype=np.float64)
    v = np.array([1.0, -2.0, 1.0, 4.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 5: Same degree polynomials (len(u) == len(v))
    u = np.array([4.0, -1.0, 5.0], dtype=np.float64)
    v = np.array([2.0, 3.0, -1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": True})
    )

    # Input 6: Negative and fractional coefficients with float64
    u = np.array([-2.5, 0.0, 1.25, -3.75, 4.0], dtype=np.float64)
    v = np.array([-1.5, 0.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 7: Complex numbers (complex64)
    u = np.array([1.0 + 2.0j, -3.0 + 0.0j, 0.5 - 1.0j], dtype=np.complex64)
    v = np.array([1.0 - 1.0j, 2.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 8: Complex numbers (complex128) with trim_leading_zeros=True
    u = np.array([2.0 - 1.5j, 4.0 + 3.0j, -1.0 + 0.0j, 1.0j], dtype=np.complex128)
    v = np.array([1.0 + 1.0j, -2.0 + 0.0j], dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": True})
    )

    # Input 9: Single-element arrays (constants)
    u = np.array([8.0], dtype=np.float32)
    v = np.array([2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": False})
    )

    # Input 10: Higher degree polynomial with int64 inputs
    u = np.array([1, 0, -4, 0, 3, 2, -1], dtype=np.int64)
    v = np.array([1, 0, -1], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": True})
    )

    # Input 11: Exact division with remainder zero
    u = np.array([1.0, -5.0, 6.0], dtype=np.float32)
    v = np.array([1.0, -2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"u": u, "v": v, "trim_leading_zeros": True})
    )

    return list_of_inputs


generated_inputs["jax.numpy.polydiv"] = polydiv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polydiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polydiv'.")


check_valid('jax.numpy.polydiv', generated_inputs['jax.numpy.polydiv'], lib="jax", suffix=0)
