
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def elu_inputs():
    list_of_inputs = []

    # Input 1: 1D array with scalar alpha (float32)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 2: 2D array with float64 precision and different scalar alpha
    x = np.array([[-3.5, 2.1], [0.0, -0.5]], dtype=np.float64)
    alpha = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 3: 3D array with float16
    x = np.random.randn(2, 3, 4).astype(np.float16)
    alpha = np.array(1.5, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 4: 0D scalar arrays
    x = np.array(-1.5, dtype=np.float32)
    alpha = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 5: Broadcastable 1D alpha with 2D x
    x = np.random.randn(4, 5).astype(np.float32)
    alpha = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 6: Higher-dimensional (4D) tensor with matching shape alpha
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    alpha = np.full((2, 2, 3, 3), 1.67326, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 7: All strictly negative values
    x = -np.abs(np.random.randn(3, 3)).astype(np.float32) - 0.1
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 8: All strictly positive values
    x = np.abs(np.random.randn(3, 3)).astype(np.float32) + 0.1
    alpha = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 9: Large negative values to check saturation behavior
    x = np.array([-100.0, -50.0, -10.0, 0.0, 10.0], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 10: Broadcastable column-vector alpha (shape (3, 1)) with 2D x (shape (3, 4))
    x = np.random.randn(3, 4).astype(np.float64)
    alpha = np.array([[0.5], [1.0], [1.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 11: Empty array edge case
    x = np.array([], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    return list_of_inputs


generated_inputs["jax.nn.elu_2"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.elu_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.elu_2'.")


check_valid('jax.nn.elu', generated_inputs['jax.nn.elu_2'], lib="jax", suffix=2)
