
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def acos_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Float32 1D array with decimals
    x = np.array([-0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Float32 2D array
    x = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Float64 1D array
    x = np.array([0.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Float64 1D array near boundaries
    x = np.array([-0.9, 0.9], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Float32 3D array
    x = np.array([[[0.5]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Float32 negative values
    x = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Float32 positive values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Float64 2D array
    x = np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Float32 1D array with 0.75 values
    x = np.array([-0.75, 0.75], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.acos'.")


check_valid('jax.lax.acos', generated_inputs['jax.lax.acos'], lib="jax", suffix=0)
