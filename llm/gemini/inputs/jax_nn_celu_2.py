
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def celu_inputs():
    list_of_inputs = []

    # Input 1: 1D array float32, standard alpha
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 2: 2D array float32, alpha=0.5
    x = np.random.randn(3, 4).astype(np.float32)
    alpha = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 3: 3D array float64, alpha=2.0
    x = np.random.randn(2, 3, 5).astype(np.float64)
    alpha = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 4: 0D array (scalar tensor), alpha=1.5
    x = np.array(-0.75, dtype=np.float32)
    alpha = 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 5: 4D array float32, alpha=0.1
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    alpha = 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 6: 1D array all non-positive, float32, alpha=1.0
    x = np.array([-10.0, -5.0, -1.0, 0.0], dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 7: 1D array all positive, float32, alpha=3.0
    x = np.array([0.1, 1.0, 5.0, 10.0], dtype=np.float32)
    alpha = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 8: 2D array with zeros, float32, alpha=0.8
    x = np.zeros((4, 4), dtype=np.float32)
    alpha = 0.8
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 9: 5D array, float32, alpha=1.2
    x = np.random.randn(1, 2, 1, 3, 4).astype(np.float32)
    alpha = 1.2
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 10: 2D array float16, alpha=0.25
    x = np.random.randn(5, 5).astype(np.float16)
    alpha = 0.25
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 11: 1D array float64 with large values, alpha=10.0
    x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    alpha = 10.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    return list_of_inputs


generated_inputs["jax.nn.celu_2"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.celu_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.celu_2'.")


check_valid('jax.nn.celu', generated_inputs['jax.nn.celu_2'], lib="jax", suffix=2)
