
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def celu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 0D scalar alpha (float32)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 2: 2D array, 0D scalar alpha (float32)
    x = np.random.randn(3, 4).astype(np.float32)
    alpha = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 3: 3D array, 0D scalar alpha (float64)
    x = np.random.randn(2, 3, 5).astype(np.float64)
    alpha = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 4: 0D scalar array x, 0D scalar alpha (float32)
    x = np.array(-0.5, dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 5: 1D array with float16, 0D float16 alpha
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float16)
    alpha = np.array(0.5, dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 6: 2D array, 1D broadcastable alpha
    x = np.random.randn(4, 5).astype(np.float32)
    alpha = np.ones((5,), dtype=np.float32) * 1.2
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 7: 4D array (e.g. NCHW), 0D alpha
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    alpha = np.array(0.8, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 8: Same shape alpha and x (2D)
    x = np.random.randn(3, 3).astype(np.float32)
    alpha = np.full((3, 3), 1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 9: 3D array, broadcastable alpha along channels (1, C, 1)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    alpha = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32).reshape(1, 4, 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 10: Array with all negative elements
    x = -np.abs(np.random.randn(5, 5).astype(np.float64)) - 0.1
    alpha = np.array(1.0, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    # Input 11: Array with all positive elements
    x = np.abs(np.random.randn(4, 6).astype(np.float32)) + 0.1
    alpha = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "alpha": copy.deepcopy(alpha)})

    return list_of_inputs


generated_inputs["jax.nn.celu_1"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.celu_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.celu_1'.")


check_valid('jax.nn.celu', generated_inputs['jax.nn.celu_1'], lib="jax", suffix=1)
