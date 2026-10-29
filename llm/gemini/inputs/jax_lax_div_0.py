
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def div_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([10.0, -20.0, 30.5, -40.2], dtype=np.float32)
    y = np.array([2.0, 4.0, -5.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 arrays with broadcasting
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D int32 arrays
    x = np.array([[10, -15], [20, -25]], dtype=np.int32)
    y = np.array([[3, 4], [-2, 5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float64 arrays
    x = np.ones((2, 3, 4), dtype=np.float64) * 12.0
    y = np.full((2, 3, 4), 3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D int64 arrays
    x = np.array(42, dtype=np.int64)
    y = np.array(-7, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D int32 arrays
    x = np.array([100, -200, 300, -400], dtype=np.int32)
    y = np.array([10, 25, -50, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D float32 arrays with broadcasting
    x = np.random.uniform(1.0, 10.0, size=(2, 1, 4, 1)).astype(np.float32)
    y = np.random.uniform(1.0, 5.0, size=(1, 3, 1, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D float64 arrays
    x = np.array([[1.5, -3.5], [7.0, -8.5]], dtype=np.float64)
    y = np.array([[0.5, 2.0], [-1.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 1D int64 arrays with negative numbers
    x = np.array([-100, -50, 0, 50, 100], dtype=np.int64)
    y = np.array([5, -5, 2, -2, 10], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D int32 arrays with broadcasting
    x = np.array([[[10, 20, 30]], [[40, 50, 60]]], dtype=np.int32)
    y = np.array([[[2, 5, 3]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.div"] = div_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.div'.")


check_valid('jax.lax.div', generated_inputs['jax.lax.div'], lib="jax", suffix=0)
