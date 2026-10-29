
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def soft_sign_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float32 array
    x = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 3D float64 array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar array
    x = np.array(-2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D float16 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array with large values
    x = np.array([-1e6, -100.0, 0.0, 100.0, 1e6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with values close to zero
    x = np.array([-1e-5, -1e-8, 0.0, 1e-8, 1e-5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D integer array (int32)
    x = np.array([[-5, -1, 0], [1, 5, 10]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D float32 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array of zeros
    x = np.zeros((10,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.soft_sign"] = soft_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.soft_sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.soft_sign'.")


check_valid('jax.nn.soft_sign', generated_inputs['jax.nn.soft_sign'], lib="jax", suffix=0)
