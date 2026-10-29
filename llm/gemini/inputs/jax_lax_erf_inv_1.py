
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erf_inv_inputs():
    list_of_inputs = []

    # Input 1: 1D array with values in (-1, 1) float32
    x = np.array([-0.8, -0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array float64
    x = np.array([[-0.9, -0.1], [0.1, 0.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array) float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array float32 with zeros and small values
    x = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array float16
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Random uniform in (-1, 1) 4D array float32
    x = np.random.uniform(-0.99, 0.99, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Single element 1D array with negative value
    x = np.array([-0.999], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D array with boundary values -1.0 and 1.0
    x = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Non-contiguous slice of array float64
    full_arr = np.linspace(-0.9, 0.9, 20, dtype=np.float64)
    x = full_arr[::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array float32
    x = np.full((1, 2, 1, 3, 1), 0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.erf_inv_1"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_1'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_1'], lib="jax", suffix=1)
