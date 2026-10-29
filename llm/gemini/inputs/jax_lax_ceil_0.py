
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ceil_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 array
    x = np.array(3.7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive and negative floats
    x = np.array([-2.7, -1.2, -0.0, 0.0, 0.1, 1.5, 2.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.array(
        [[-1.5, 2.3, -0.7], [4.1, -5.9, 0.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D float16 array
    x = np.array([-10.5, -0.5, 0.5, 10.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array random values
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array with negative numbers only
    x = -np.random.exponential(scale=2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with special values (inf, -inf, nan)
    x = np.array([-np.inf, np.inf, np.nan, -1.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D array float64
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array with exact integers represented as floats
    x = np.array([[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with very small fractional parts
    x = np.array([1.0000001, -1.0000001, 1e-7, -1e-7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ceil'.")


check_valid('jax.lax.ceil', generated_inputs['jax.lax.ceil'], lib="jax", suffix=0)
