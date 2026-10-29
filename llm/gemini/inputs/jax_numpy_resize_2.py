
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def resize_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array to 2D shape (repeating elements)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    new_shape = (3, 4)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 2: 1D integer array to exact 2D shape
    a = np.arange(9, dtype=np.int64)
    new_shape = (3, 3)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 3: 2D float32 array truncated to smaller 1D shape
    a = np.array([[-1.5, 2.3, 0.0], [4.1, -5.2, 6.8]], dtype=np.float32)
    new_shape = (4,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 4: 0D scalar array resized to 2D
    a = np.array(-7, dtype=np.int32)
    new_shape = (3, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 5: Boolean array resized to 2D
    a = np.array([True, False, True], dtype=np.bool_)
    new_shape = (2, 4)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 6: Float64 array resized to 3D
    a = np.linspace(-10.0, 10.0, 5, dtype=np.float64)
    new_shape = (2, 2, 3)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 7: Complex array resized to 2D
    a = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
    new_shape = (3, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 8: 3D float32 array flattened to 1D
    a = np.ones((2, 3, 4), dtype=np.float32)
    new_shape = (10,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 9: Unsigned integer array resized to 4D
    a = np.array([10, 20, 30], dtype=np.uint8)
    new_shape = (1, 2, 2, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 10: Array resized to a 0D scalar
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    new_shape = ()
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    # Input 11: Large 1D array resized to a multi-dimensional array
    a = np.arange(100, dtype=np.int16)
    new_shape = (5, 5, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "new_shape": new_shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.resize_2"] = resize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.resize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.resize_2'.")


check_valid('jax.numpy.resize', generated_inputs['jax.numpy.resize_2'], lib="jax", suffix=2)
