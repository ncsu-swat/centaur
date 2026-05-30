
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def relu6_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array spanning negative, within-range, and above-6 values
    x = np.array([-2.0, -0.5, 0.0, 3.0, 5.9, 6.0, 7.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 4D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: float64 1D array
    x = np.array([-1.5, 0.0, 2.5, 6.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: float16 2D array
    x = np.random.uniform(-5.0, 8.0, size=(3, 5)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: int32 1D array
    x = np.array([-5, -1, 0, 3, 6, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: int64 2D array
    x = np.random.randint(-10, 10, size=(2, 4)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 0D array (scalar tensor)
    x = np.array(4.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float32 array
    x = np.random.uniform(-2.0, 8.0, size=(1, 2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Array containing only values below the threshold (all negative)
    x = np.random.uniform(-20.0, -1.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.relu6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.relu6'.")


check_valid('jax.nn.relu6', generated_inputs['jax.nn.relu6'], lib="jax", suffix=0)
