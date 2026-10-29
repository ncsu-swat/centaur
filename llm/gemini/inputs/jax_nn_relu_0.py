
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def relu_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative, zero, and positive values
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float32 array
    x = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor float32
    x = np.array(-3.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float64 array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D float32 array (e.g. NCHW format)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D int32 array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D int64 array
    x = np.array([[-3, 0], [4, -1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: float16 array
    x = np.array([-1.5, 0.0, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with infinity and NaN values
    x = np.array([-np.inf, -1.0, 0.0, np.nan, 1.0, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D float32 array
    x = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: All negative values
    x = np.array([-100.0, -50.0, -1e-5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.relu"] = relu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.relu'.")


check_valid('jax.nn.relu', generated_inputs['jax.nn.relu'], lib="jax", suffix=0)
