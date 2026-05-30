
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hard_silu_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with boundary values around hard sigmoid thresholds (-3.0 and 3.0)
    x = np.array([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with random values
    x = np.random.uniform(-5.0, 5.0, size=(3, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Scalar (0D array) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float32 array with saturated values
    x = np.array([[[[-100.0]], [[0.0]], [[100.0]]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float16 array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float32 array containing special values (infinity, NaN)
    x = np.array([[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: High-dimensional (5D) float32 array
    x = np.random.normal(0, 1, size=(2, 2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D float32 array of all zeros
    x = np.zeros((10,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array of large positive values
    x = np.ones((5,), dtype=np.float32) * 10.0
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.hard_silu"] = hard_silu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_silu'.")


check_valid('jax.nn.hard_silu', generated_inputs['jax.nn.hard_silu'], lib="jax", suffix=0)
