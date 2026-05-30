
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sigmoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard normal
    x = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array, positive values
    x = np.random.uniform(0.1, 10.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array, negative values
    x = np.random.uniform(-10.0, -0.1, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D float64 array, high precision
    x = np.random.randn(100).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float16 array, low precision
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar shape)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Array with large values (testing saturation to 1.0)
    x = np.array([100.0, 50.0, 1000.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array with highly negative values (testing saturation to 0.0)
    x = np.array([-100.0, -50.0, -1000.0, -20.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D float32 array, high dimension
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.sigmoid"] = sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sigmoid'.")


check_valid('jax.nn.sigmoid', generated_inputs['jax.nn.sigmoid'], lib="jax", suffix=0)
