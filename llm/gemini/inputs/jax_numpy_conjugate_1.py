
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def conjugate_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j, 5.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D complex128 array
    x = np.array(
        [[1.5 - 2.5j, -0.5 + 4.2j], [3.1 + 0.0j, -2.2 - 1.1j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor (complex64)
    x = np.array(2.0 - 7.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D float32 array (real values)
    x = np.array([-1.0, 0.0, 2.5, -3.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float64 array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D integer array (int32)
    x = np.array([[-5, 0, 3], [12, -8, 7]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D complex64 array
    real_part = np.random.randn(2, 2, 3, 2).astype(np.float32)
    imag_part = np.random.randn(2, 2, 3, 2).astype(np.float32)
    x = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D boolean array
    x = np.array([True, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.empty((0, 3), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with special values (inf, nan)
    x = np.array(
        [complex(np.nan, 1.0), complex(np.inf, -2.0), complex(-np.inf, np.nan)],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D int64 array
    x = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 10
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.conjugate_1"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_1'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_1'], lib="jax", suffix=1)
