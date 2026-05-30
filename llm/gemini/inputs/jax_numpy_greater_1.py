
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D float arrays of same shape
    x = np.array([1.5, -2.3, 0.0, 5.7], dtype=np.float32)
    y = np.array([2.0, -3.0, 0.0, 1.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D int arrays of same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[0, 2], [4, 1]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Broadcast 2D and 1D arrays
    x = np.array([[5, -6, 7], [-2, 5, 9]], dtype=np.float32)
    y = np.array([-4, 3, 10], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcast 1D and 2D arrays (reversed roles)
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([[1], [2], [3]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 3D float64 arrays of same shape
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D arrays (scalars wrapped in arrays)
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Boolean arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, False, True, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: High dimensional broadcast
    x = np.random.rand(1, 4, 1, 5).astype(np.float32)
    y = np.random.rand(3, 1, 2, 5).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 1D int32 arrays with negative and positive values
    x = np.array([10, -20, 30], dtype=np.int32)
    y = np.array([15, -15, 15], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D float64 arrays with various ranges
    x = np.array([-10.0, 10.0, 0.5], dtype=np.float64)
    y = np.array([-5.0, 5.0, 0.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.greater_1"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_1'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_1'], lib="jax", suffix=1)
