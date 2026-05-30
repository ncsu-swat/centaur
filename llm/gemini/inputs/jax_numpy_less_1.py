
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def less_inputs():
    list_of_inputs = []

    # Input 1: Same shape, 1D, integers with negative values
    x = np.array([1, -2, 3, 0], dtype=np.int32)
    y = np.array([2, -1, 3, -1], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: Same shape, 1D, floats with negative values
    x = np.array([1.5, -2.5, 0.0], dtype=np.float32)
    y = np.array([1.4, -2.0, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Same shape, 2D, float64
    x = np.random.randn(3, 3).astype(np.float64)
    y = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcasting, 2D and 1D
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Broadcasting, 2D row vector and column vector
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[2, 3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D arrays (scalars wrapped in arrays)
    x = np.array(5, dtype=np.int32)
    y = np.array(10, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 3D arrays, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Boolean arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, False, True, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 4D arrays, broadcasting
    x = np.ones((2, 1, 3, 1), dtype=np.float32)
    y = np.zeros((1, 4, 1, 5), dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Larger 1D arrays
    x = np.arange(100).astype(np.int32)
    y = np.arange(100).astype(np.int32) + 1
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.less_1"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_1'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_1'], lib="jax", suffix=1)
