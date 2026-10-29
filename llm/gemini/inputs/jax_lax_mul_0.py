
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_mul_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([-1.5, 2.0, 0.0, -4.5], dtype=np.float32)
    y = np.array([2.0, -3.0, 5.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 arrays with identical shapes
    x = np.array([[1, -2], [3, -4]], dtype=np.int32)
    y = np.array([[-5, 6], [-7, 8]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D scalar float32 arrays
    x = np.array(-3.14, dtype=np.float32)
    y = np.array(2.71, dtype=np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D broadcastable shapes (1, 4) and (3, 4) float32
    x = np.full((1, 4), 2.5, dtype=np.float32)
    y = np.full((3, 4), -1.2, dtype=np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 arrays with negative values
    x = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    y = np.full((2, 3, 4), 3, dtype=np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 arrays with zeros and special values
    x = np.array([[-0.0, 10.5], [np.nan, -np.inf]], dtype=np.float32)
    y = np.array([[2.0, -1.0], [3.0, 0.5]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D scalar int32 arrays
    x = np.array(42, dtype=np.int32)
    y = np.array(-7, dtype=np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 arrays
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 arrays with broadcasting along dimension 1
    x = np.ones((2, 1, 3), dtype=np.int32) * 5
    y = np.ones((2, 4, 3), dtype=np.int32) * 3
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int32 arrays casting to float32
    x = np.array([10, -20, 30, 0], dtype=np.int32)
    y = np.array([-2, 3, -4, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "out_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.mul"] = lax_mul_inputs()

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
