
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_sub_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays of the same shape
    x = np.random.randn(5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float32 arrays with negative values
    x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    y = np.array([[0.5, -0.5, 1.5], [-1.5, 2.5, -2.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D int32 arrays of the same shape
    x = np.random.randint(-10, 10, size=(2, 3, 2)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 3, 2)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D int64 arrays of the same shape
    x = np.random.randint(-100, 100, size=(10,)).astype(np.int64)
    y = np.random.randint(-100, 100, size=(10,)).astype(np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D float64 arrays of the same shape
    x = np.random.randn(4, 4).astype(np.float64)
    y = np.random.randn(4, 4).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D arrays (scalars represented as numpy arrays)
    x = np.array(5.5, dtype=np.float32)
    y = np.array(2.3, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Broadcasting with same number of dimensions (2D), y has a dimension of 1
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(1, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Broadcasting with same number of dimensions (3D), mixed singleton dimensions
    x = np.random.randn(2, 1, 4).astype(np.float32)
    y = np.random.randn(2, 3, 1).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large 2D float32 arrays
    x = np.random.randn(128, 128).astype(np.float32)
    y = np.random.randn(128, 128).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: float16 arrays of the same shape
    x = np.random.randn(5, 5).astype(np.float16)
    y = np.random.randn(5, 5).astype(np.float16)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.sub_1"] = jax_lax_sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_1'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_1'], lib="jax", suffix=1)
