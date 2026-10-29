
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hard_swish_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative and positive values
    x = np.array([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D scalar float32 array
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float64 array (matrix)
    x = np.linspace(-6.0, 6.0, 12, dtype=np.float64).reshape(3, 4)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array representing a batch of sequences
    x = np.random.randn(2, 5, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float32 array representing NCHW image tensors
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 16, 16)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float16 array
    x = np.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 5D float32 array
    x = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large magnitude values in float32
    x = np.array([-1e5, -100.0, 0.0, 100.0, 1e5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D integer array (jax handles integer array input by casting or elementwise ops)
    x = np.array([[-5, -3, 0], [1, 3, 5]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 2D float32 array with all zeros
    x = np.zeros((4, 4), dtype=np.float32)
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
