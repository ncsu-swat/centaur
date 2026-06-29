
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def neg_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mixed positive and negative values
    x = np.array([-2.5, 0.0, 3.14, -100.0, 42.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array
    x = np.random.randint(-100, 100, size=(2, 3, 2)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int64 array
    x = np.array([-922337203685477580, 0, 922337203685477580], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Scalar (0D array) float32
    x = np.array(-5.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D int32 array
    x = np.random.randint(-50, 50, size=(3, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float64 array
    x = np.array([-1e10, 0.0, 1e10], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D int64 array
    x = np.random.randint(-500, 500, size=(1, 2, 2, 1, 3)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Large 1D float32 array
    x = np.random.randn(1000).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.neg"] = neg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.neg'.")


check_valid('jax.lax.neg', generated_inputs['jax.lax.neg'], lib="jax", suffix=0)
