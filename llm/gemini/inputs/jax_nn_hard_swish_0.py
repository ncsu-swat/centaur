
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hard_swish_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with typical small values
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Scalar-like 0D array
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: float64 precision 2D array
    x = np.random.randn(4, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: float16 precision 2D array
    x = np.random.randn(5, 2).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large values where hard_sigmoid saturates (x <= -3 and x >= 3)
    x = np.array([-10.0, -5.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D tensor representing a batch of images
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with all zeros
    x = np.zeros((10,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array with highly positive values
    x = np.random.uniform(3.0, 100.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Array with highly negative values
    x = np.random.uniform(-100.0, -3.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.hard_swish"] = hard_swish_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_swish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_swish'.")


check_valid('jax.nn.hard_swish', generated_inputs['jax.nn.hard_swish'], lib="jax", suffix=0)
