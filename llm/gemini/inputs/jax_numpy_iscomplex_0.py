
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iscomplex_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array with both complex and purely real numbers
    x = np.array([1.0 + 2.0j, 3.0 + 0.0j, 0.0 + 4.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D complex128 array with negative components
    x = np.array(
        [[-1.5 - 2.5j, 3.0 + 0.0j], [0.0 - 1.0j, -4.0 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D standard float32 array (all should be non-complex)
    x = np.array([1.0, -2.5, 3.14, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D complex scalar array
    x = np.array(2.0 + 3.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D real scalar array
    x = np.array(42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D integer array
    x = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty complex array
    x = np.array([], dtype=np.complex128).reshape((0, 5))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D complex array
    real_part = np.random.randn(2, 2, 3, 2).astype(np.float32)
    imag_part = np.random.randn(2, 2, 3, 2).astype(np.float32)
    x = real_part + 1j * imag_part
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D complex array containing NaNs and Infs in imaginary parts
    x = np.array(
        [[1.0 + np.nan * 1j, np.inf + 2.0j], [-np.inf + 0.0j, 0.0 + np.inf * 1j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D int64 array with zeros and negative values
    x = np.array([-10, 0, 10, -100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.iscomplex"] = iscomplex_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplex'.")


check_valid('jax.numpy.iscomplex', generated_inputs['jax.numpy.iscomplex'], lib="jax", suffix=0)
