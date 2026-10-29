
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def real_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    val = np.array([1.0 + 2.0j, -3.5 + 0.0j, 0.0 - 4.2j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 2: 2D complex128 array with positive and negative components
    val = np.array(
        [[1.5 - 2.5j, -3.0 + 4.0j], [0.0 + 0.0j, -5.5 - 6.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 3: 0D scalar complex array
    val = np.array(3.0 - 7.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 4: 1D float32 array (real inputs are valid)
    val = np.array([-1.2, 0.0, 3.4, 5.6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 5: 2D int32 array
    val = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 6: 3D complex64 array
    val = (
        np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 7: 4D float64 array
    val = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 8: Empty 1D array
    val = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 9: 2D boolean array
    val = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 10: 1D complex128 array with special values (inf, nan)
    val = np.array(
        [np.nan + 1.0j, np.inf - 2.0j, -np.inf + 0.0j], dtype=np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 11: 3D int64 array
    val = np.arange(-12, 12, dtype=np.int64).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"val": val}))

    return list_of_inputs


generated_inputs["jax.numpy.real_1"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.real_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.real_1'.")


check_valid('jax.numpy.real', generated_inputs['jax.numpy.real_1'], lib="jax", suffix=1)
