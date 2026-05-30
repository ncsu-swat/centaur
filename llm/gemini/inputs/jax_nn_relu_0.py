
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_relu_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive, negative, and zero values (float32)
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x1)})

    # Input 2: 2D array with float64 values
    x2 = np.array([[-1.5, 0.5, -0.0], [2.3, -3.4, 1.1]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x2)})

    # Input 3: 0D array (scalar tensor)
    x3 = np.array(-5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x3)})

    # Input 4: 3D array of float16 values
    x4 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x4)})

    # Input 5: 4D array mimicking a batch of images (NCHW)
    x5 = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x5)})

    # Input 6: 1D array with integer values
    x6 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x6)})

    # Input 7: Large 2D matrix of random normal values
    x7 = np.random.randn(128, 128).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x7)})

    # Input 8: All-zero tensor of 3D shape
    x8 = np.zeros((3, 3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x8)})

    # Input 9: 5D tensor
    x9 = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x9)})

    # Input 10: 1D array with very small/extreme values
    x10 = np.array([-1e-15, 1e-15, -1e15, 1e15], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x10)})

    # Input 11: 2D array with int16 values
    x11 = np.array([[-100, 200], [-300, 400]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x11)})

    return list_of_inputs

generated_inputs["jax.nn.relu"] = generate_relu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.relu'.")


check_valid('jax.nn.relu', generated_inputs['jax.nn.relu'], lib="jax", suffix=0)
