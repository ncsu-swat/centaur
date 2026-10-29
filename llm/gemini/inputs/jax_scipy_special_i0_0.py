
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def i0_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive values
    x = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative values (i0 is symmetric: i0(-x) = i0(x))
    x = np.array([-5.0, -2.0, -1.0, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array of zeros
    x = np.zeros((10,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float64
    x = np.random.uniform(-10.0, 10.0, size=(4, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array float32
    x = np.linspace(-3.0, 3.0, num=12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array float32
    x = np.random.uniform(0.0, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with integer values (promoted to float in JAX)
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D float16 array
    x = np.array([0.1, 0.5, 1.2, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i0'.")


check_valid('jax.scipy.special.i0', generated_inputs['jax.scipy.special.i0'], lib="jax", suffix=0)
