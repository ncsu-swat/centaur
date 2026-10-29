
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lgamma_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of positive float32 values
    x = np.array([0.5, 1.0, 1.5, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative non-integer float32 values
    x = np.array([-0.5, -1.5, -2.5, 0.1, 3.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with random positive values, float32
    x = np.random.uniform(0.1, 10.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array float32
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array float64
    x = np.random.uniform(0.5, 2.5, size=(2, 2, 3, 3)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: float16 array
    x = np.array([1.2, 2.3, 3.4, 4.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large positive values, float64
    x = np.array([10.0, 50.0, 100.0, 150.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with values near zero
    x = np.array([0.001, 0.01, 0.1, 0.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.lgamma_1"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_1'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_1'], lib="jax", suffix=1)
