
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def i0e_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of small positive float32 values
    x = np.array([0.1, 0.5, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array of negative float64 values
    x = np.array([-10.0, -5.0, -1.0, -0.01], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of mixed positive and negative float32 values
    x = np.array([[-3.0, -1.5, 0.0], [1.5, 3.0, 4.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with large values (float64)
    x = np.array([[100.0, 500.0], [-100.0, -500.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of random standard normal values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array of small uniform float32 values
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array of integer values
    x = np.array([-5, -2, 0, 1, 3, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array of zeros
    x = np.zeros((4, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array of float16 values
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i0e'.")


check_valid('jax.scipy.special.i0e', generated_inputs['jax.scipy.special.i0e'], lib="jax", suffix=0)
