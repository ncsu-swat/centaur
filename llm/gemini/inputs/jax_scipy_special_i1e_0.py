
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def i1e_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of zeros and small values
    x = np.array([0.0, 0.1, -0.1, 0.5, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array float64 with negative values
    x = np.array([-10.0, -5.0, -1.0, -0.01], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of random standard normal values
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float64 with large values (where exponential scaling prevents overflow)
    x = np.array([[50.0, 100.0], [-50.0, -100.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of float32
    x = np.linspace(-5.0, 5.0, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D integer array
    x = np.array([-5, -1, 0, 1, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array float32
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array float16
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D integer array (int64)
    x = np.array([[-10, 0, 10], [20, -20, 5]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.i1e"] = i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i1e'.")


check_valid('jax.scipy.special.i1e', generated_inputs['jax.scipy.special.i1e'], lib="jax", suffix=0)
