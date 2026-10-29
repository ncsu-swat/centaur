
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erfcx_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values including positive, negative, and zero
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of float64 values with large positive numbers where erfcx is stable
    x = np.array([5.0, 10.0, 25.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float32 values
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of float64 values
    x = np.random.uniform(-3.0, 3.0, size=(5, 2)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of float32 values
    x = np.linspace(-4.0, 4.0, num=24, dtype=np.float32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of float64 values
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 0D array (scalar tensor)
    x = np.array(1.75, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (should be promoted to floating point by JAX)
    x = np.array([-5, -2, 0, 3, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with small values close to zero
    x = np.array([-1e-5, -1e-8, 0.0, 1e-8, 1e-5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing infinity
    x = np.array([np.inf, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array of float16 values
    x = np.array([-1.5, 0.5, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.erfcx"] = erfcx_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erfcx' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erfcx'.")


check_valid('jax.scipy.special.erfcx', generated_inputs['jax.scipy.special.erfcx'], lib="jax", suffix=0)
