
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([1, 0, 3, 0, 5], dtype=np.int32)
    axis = (0,)
    keepdims = False
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 2: 2D integer array with keepdims=True
    a = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.int64)
    axis = (0,)
    keepdims = True
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 3: 2D float array with some zeros
    a = np.random.randn(3, 4).astype(np.float32)
    a[a < 0] = 0.0
    axis = (1,)
    keepdims = False
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 4: 3D boolean array counting over multiple axes
    a = np.random.choice([True, False], size=(2, 3, 4))
    axis = (0, 2)
    keepdims = True
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 5: 3D complex array
    a = np.array([[[1 + 1j, 0], [0, 2j]]], dtype=np.complex64)
    axis = (1,)
    keepdims = False
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 6: 4D integer array
    a = np.random.randint(-5, 5, size=(2, 2, 3, 3)).astype(np.int32)
    axis = (2, 3)
    keepdims = True
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 7: 2D float64 array with negative values
    a = np.array([[-1.0, 0.0], [2.5, -3.2]], dtype=np.float64)
    axis = (1,)
    keepdims = True
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 8: 1D array of all zeros
    a = np.zeros((10,), dtype=np.int16)
    axis = (0,)
    keepdims = False
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 9: 3D array of all ones (no zeros)
    a = np.ones((2, 3, 2), dtype=np.float32)
    axis = (0, 1)
    keepdims = False
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    # Input 10: 5D high-dimensional float array
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    keepdims = True
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims})

    return list_of_inputs

generated_inputs["jax.numpy.count_nonzero_2"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_2'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_2'], lib="jax", suffix=2)
