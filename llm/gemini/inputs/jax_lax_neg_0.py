
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_neg_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-2.5, 0.0, 3.5, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array with positive and negative values
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array) int32
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D int64 array
    x = np.arange(-12, 12, dtype=np.int64).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D int32 array
    x = np.array([[10, -20], [-30, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Complex64 1D array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Complex128 2D array
    x = np.array(
        [[2.5 - 1.5j, -4.0 + 0.0j], [0.0 + 0.0j, -1.2 - 3.4j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float16 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D float32 array with inf and nan
    x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array (0 elements) float32
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.neg"] = lax_neg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.neg'.")


check_valid('jax.lax.neg', generated_inputs['jax.lax.neg'], lib="jax", suffix=0)
