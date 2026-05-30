
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def swish_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive and negative float32 values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array with float32 values (standard matrix)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array with float64 values (double precision)
    x = np.random.randn(2, 3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar) of float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D array of float32 (e.g. batch of images NCHW)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with float16 (half precision)
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large 2D array with float32
    x = np.random.randn(128, 128).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Array containing large magnitude values to test numerical limits
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with small values close to zero
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array of float32
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: All zeros array
    x = np.zeros((4, 4), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.swish"] = swish_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.swish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.swish'.")


check_valid('jax.nn.swish', generated_inputs['jax.nn.swish'], lib="jax", suffix=0)
