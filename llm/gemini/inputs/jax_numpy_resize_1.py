
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def resize_inputs():
    list_of_inputs = []

    # Input 1: 1D array, larger new_shape
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    new_shape = 10
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 2: 2D array, smaller new_shape
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    new_shape = 3
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 3: Float32 array, 3D
    a = np.random.randn(2, 3, 2).astype(np.float32)
    new_shape = 12
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 4: Float64 array, 1D
    a = np.linspace(-1.0, 1.0, 5).astype(np.float64)
    new_shape = 8
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 5: Large int32 array, many dimensions
    a = np.arange(16).reshape(2, 2, 2, 2).astype(np.int32)
    new_shape = 5
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 6: Boolean array
    a = np.array([True, False, True], dtype=bool)
    new_shape = 6
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 7: Float32 array, 2D
    a = np.random.randn(3, 3).astype(np.float32)
    new_shape = 4
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 8: Int64 array
    a = np.array([5, 10, 15], dtype=np.int64)
    new_shape = 4
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 9: Int32 array
    a = np.array([[10, 20], [30, 40]], dtype=np.int32)
    new_shape = 8
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    # Input 10: 1-element float32
    a = np.array([3.14], dtype=np.float32)
    new_shape = 5
    list_of_inputs.append({"a": a, "new_shape": new_shape})

    return list_of_inputs

generated_inputs["jax.numpy.resize_1"] = resize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.resize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.resize_1'.")


check_valid('jax.numpy.resize', generated_inputs['jax.numpy.resize_1'], lib="jax", suffix=1)
