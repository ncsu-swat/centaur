
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rem_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([10, -10, 7, -7], dtype=np.int32)
    y = np.array([3, 3, -3, -3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D float32 arrays
    x = np.array([[5.5, -5.5], [12.2, -12.2]], dtype=np.float32)
    y = np.array([[2.0, 2.0], [3.0, -3.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 3D float64 arrays
    x = np.random.uniform(10.0, 50.0, size=(2, 3, 4)).astype(np.float64)
    y = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 1D int64 arrays with negative values
    x = np.array([-100, 250, -350, 400], dtype=np.int64)
    y = np.array([7, -13, 29, -31], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 2D int32 arrays with broadcasting
    x = np.array([[15, 25, 35], [45, 55, 65]], dtype=np.int32)
    y = np.array([[4], [6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 0D scalar float32 arrays
    x = np.array(7.5, dtype=np.float32)
    y = np.array(2.25, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 4D float32 arrays
    x = np.random.uniform(-20.0, 20.0, size=(2, 2, 3, 3)).astype(np.float32)
    y = np.random.uniform(1.5, 4.5, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 2D int64 arrays
    x = np.array([[100, 200], [300, 400]], dtype=np.int64)
    y = np.array([[7, 13], [17, 19]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 1D float32 arrays
    x = np.array([12.5, -25.2, 64.1, -120.8], dtype=np.float32)
    y = np.array([5.1, 4.2, 7.3, -9.4], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 3D int32 arrays with broadcasting
    x = np.array([[[10, 20, 30]], [[40, 50, 60]]], dtype=np.int32)
    y = np.array([[[3, 7, 9]], [[4, 6, 8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 2D float64 arrays with broadcasting
    x = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
    y = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs


generated_inputs["jax.lax.rem"] = rem_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rem' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rem'.")


check_valid('jax.lax.rem', generated_inputs['jax.lax.rem'], lib="jax", suffix=0)
