
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ldexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive exponent
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float32 array, negative exponent
    x1 = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    x2 = -2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 1D float64 array, zero exponent
    x1 = np.array([-1.5, 0.0, 2.5, 10.0], dtype=np.float64)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 3D float32 array, positive exponent
    x1 = np.ones((2, 3, 4), dtype=np.float32)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 0D scalar array float32, positive exponent
    x1 = np.array(3.5, dtype=np.float32)
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 2D float64 array with negative numbers, negative exponent
    x1 = np.array([[-10.0, -20.0], [-30.0, -40.0]], dtype=np.float64)
    x2 = -3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array, small positive exponent
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float64 array with subnormals/small values, positive exponent
    x1 = np.array([1e-10, 1e-5, 1e-2], dtype=np.float64)
    x2 = 10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float32 array, larger negative exponent
    x1 = np.array([[1024.0, 2048.0], [4096.0, 8192.0]], dtype=np.float32)
    x2 = -10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D float32 array with mixed signs and zeros, positive exponent
    x1 = np.array([-5.0, -0.0, 0.0, 5.0], dtype=np.float32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D float64 array, negative exponent
    x1 = np.full((3, 2, 1), 64.0, dtype=np.float64)
    x2 = -6
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.ldexp_2"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ldexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ldexp_2'.")


check_valid('jax.numpy.ldexp', generated_inputs['jax.numpy.ldexp_2'], lib="jax", suffix=2)
