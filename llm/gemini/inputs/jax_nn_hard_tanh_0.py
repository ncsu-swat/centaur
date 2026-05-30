
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hard_tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array with float32, crossing boundaries [-2, 2]
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array, float32, uniform random in [-2.0, 2.0]
    x = np.random.uniform(-2.0, 2.0, size=(3, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array, float64
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar as tensor)
    x = np.array(-1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D array, float16
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array, small values close to zero
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array, all values > 1 (should saturate to 1)
    x = np.random.uniform(1.1, 10.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D array, all values < -1 (should saturate to -1)
    x = np.random.uniform(-10.0, -1.1, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array of size 1
    x = np.array([0.75], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array, float32, containing random values
    x = np.random.uniform(-1.5, 1.5, size=(1, 2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.hard_tanh"] = hard_tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_tanh'.")


check_valid('jax.nn.hard_tanh', generated_inputs['jax.nn.hard_tanh'], lib="jax", suffix=0)
