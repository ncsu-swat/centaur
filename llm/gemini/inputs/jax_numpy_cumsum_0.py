
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, axis 0
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = 0
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 2: 2D float32 array, axis 1
    a = np.array([[1.0, -2.0, 3.5], [-4.0, 5.2, -6.1]], dtype=np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 3: 2D float64 array, axis 0
    a = np.random.randn(4, 5).astype(np.float64)
    axis = 0
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 4: 2D int32 array accumulated along negative axis -1 to int64
    a = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int32)
    axis = -1
    dtype = np.dtype(np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 5: 3D float32 array, axis 2
    a = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    axis = 2
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 6: 3D float32 array, negative axis -2, casting to float64
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = -2
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 7: 1D boolean array, axis 0, casting to int32
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    axis = 0
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 8: 4D float32 array, axis 1
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 9: 2D complex64 array, axis 0
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    axis = 0
    dtype = np.dtype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 10: 1D float32 array with negative axis -1
    a = np.array([-1.5, -2.5, -3.5, 0.0, 4.5], dtype=np.float32)
    axis = -1
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    # Input 11: 3D uint8 array accumulated along axis 0 into uint32
    a = np.random.randint(0, 255, size=(4, 2, 2), dtype=np.uint8)
    axis = 0
    dtype = np.dtype(np.uint32)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis, "dtype": dtype}))

    return list_of_inputs


generated_inputs["jax.numpy.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumsum'.")


check_valid('jax.numpy.cumsum', generated_inputs['jax.numpy.cumsum'], lib="jax", suffix=0)
