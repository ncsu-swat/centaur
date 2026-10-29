
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_eq_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, 2.0, -3.0, 4.5], dtype=np.float32)
    y = np.array([1.0, -2.0, -3.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D int32 arrays of same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 0], [3, -4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 2D float64 with broadcasting (same number of dimensions)
    x = np.ones((3, 4), dtype=np.float64)
    y = np.ones((3, 1), dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 0D scalar arrays (matching dtypes)
    x = np.array(5, dtype=np.int64)
    y = np.array(5, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 3D int16 arrays with broadcasting along multiple dimensions
    x = np.zeros((2, 1, 4), dtype=np.int16)
    y = np.zeros((1, 3, 4), dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 4D float32 arrays
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    y = x.copy()
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Complex64 arrays
    x = np.array([1.0 + 2.0j, -3.0 + 0.0j, 0.5 - 1.5j], dtype=np.complex64)
    y = np.array([1.0 + 2.0j, -3.0 + 1.0j, 0.5 - 1.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: uint8 arrays
    x = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    y = np.array([[0, 200], [128, 32]], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Empty arrays (0-length dimension)
    x = np.empty((0, 5), dtype=np.float32)
    y = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs


generated_inputs["jax.lax.eq_1"] = jax_lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_1'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_1'], lib="jax", suffix=1)
