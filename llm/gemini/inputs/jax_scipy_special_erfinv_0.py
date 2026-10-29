
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erfinv_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with values in (-1, 1)
    x = np.array([-0.8, -0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with domain boundaries [-1, 1]
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of float32 values
    x = np.linspace(-0.95, 0.95, 12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array of float64 values
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.9, 0.9, size=(4, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of float32 values
    x = rng.uniform(-0.99, 0.99, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Values very close to zero
    x = np.array([-1e-7, -1e-5, 0.0, 1e-5, 1e-7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Float16 array
    x = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D array
    x = rng.uniform(-0.5, 0.5, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing all zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.erfinv"] = erfinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erfinv'.")


check_valid('jax.scipy.special.erfinv', generated_inputs['jax.scipy.special.erfinv'], lib="jax", suffix=0)
