
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, y is 0
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D int32 array, y is 5
    x = np.random.randint(0, 10, size=(3, 3)).astype(np.int32)
    y = 5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int32 array, y is -10
    x = np.random.randint(-20, 0, size=(2, 2, 2)).astype(np.int32)
    y = -10
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 0D int32 array, y is 42
    x = np.array(42, dtype=np.int32)
    y = 42
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D int32 array, y is 1
    x = np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.int32)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 1D int32 array, y is 100
    x = np.array([50, 100, 150], dtype=np.int32)
    y = 100
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 2D int32 array, y is -5
    x = np.random.randint(-10, 10, size=(4, 2)).astype(np.int32)
    y = -5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 3D int32 array, y is 0
    x = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int32)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 1D int32 array, y is 127
    x = np.array([-128, 0, 127], dtype=np.int32)
    y = 127
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 5D int32 array, y is -1
    x = np.random.randint(-5, 5, size=(1, 2, 1, 3, 1)).astype(np.int32)
    y = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.lax.le_6"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_6'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_6'], lib="jax", suffix=6)
