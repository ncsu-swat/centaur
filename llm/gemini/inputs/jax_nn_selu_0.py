
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def selu_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with negative and positive values
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of float32 standard normal
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 0D scalar array of float64
    x = np.array(-1.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array of float64
    x = np.random.randn(2, 3, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D array of float16
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with all zeros
    x = np.zeros((10,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array with all negative values
    x = -np.abs(np.random.randn(8).astype(np.float32)) - 0.1
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with all positive values
    x = np.abs(np.random.randn(8).astype(np.float32)) + 0.1
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D array
    x = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array with large positive and negative values
    x = np.array([[-100.0, -10.0, -1.0], [1.0, 10.0, 100.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.nn.selu"] = selu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.selu'.")


check_valid('jax.nn.selu', generated_inputs['jax.nn.selu'], lib="jax", suffix=0)
