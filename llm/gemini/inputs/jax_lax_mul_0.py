
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_mul_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.random.randn(5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 arrays with negative values
    x = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 1D arrays
    x = np.random.randn(10).astype(np.float64)
    y = np.random.randn(10).astype(np.float64)
    input_dict = {"x": x, "y": y, "out_dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcastable 2D arrays
    x = np.random.randn(4, 1).astype(np.float32)
    y = np.random.randn(1, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar) and 3D array
    x = np.array(2.5, dtype=np.float32)
    y = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int32 arrays with negative values
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int32)
    y = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Cast float32 inputs to float64 output
    x = np.random.randn(3, 3).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int32 arrays
    x = np.random.randint(-50, 50, size=(8,)).astype(np.int32)
    y = np.random.randint(-50, 50, size=(8,)).astype(np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional 5D float32 arrays
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.mul"] = jax_lax_mul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.mul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.mul'.")


check_valid('jax.lax.mul', generated_inputs['jax.lax.mul'], lib="jax", suffix=0)
