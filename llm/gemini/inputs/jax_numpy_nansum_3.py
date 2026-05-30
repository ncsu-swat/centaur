
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy

# Monkeypatch jax.numpy.nansum to support list for axis (converting to tuple internally)
_orig_nansum = jax.numpy.nansum

def patched_nansum(*args, **kwargs):
    new_args = list(args)
    if len(new_args) > 1 and isinstance(new_args[1], list):
        new_args[1] = tuple(new_args[1])
    if 'axis' in kwargs and isinstance(kwargs['axis'], list):
        kwargs['axis'] = tuple(kwargs['axis'])
    return _orig_nansum(*new_args, **kwargs)

jax.numpy.nansum = patched_nansum

def nansum_inputs():
    list_of_inputs = []

    def make_nan_array(shape, dtype):
        arr = np.random.randn(*shape).astype(dtype)
        mask = np.random.rand(*shape) < 0.2
        arr[mask] = np.nan
        return arr

    # Input 1: Simple 2D array, reduce axis 0
    a = make_nan_array((4, 5), np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = True
    initial = 0.0
    where = np.random.choice([True, False], size=(4, 5), p=[0.9, 0.1])
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reduce axis 1, float64
    a = make_nan_array((3, 6), np.float64)
    axis = [1]
    dtype = np.float64
    keepdims = False
    initial = 10.5
    where = np.ones((3, 6), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, reduce multiple axes
    a = make_nan_array((2, 3, 4), np.float32)
    axis = [0, 2]
    dtype = np.float32
    keepdims = True
    initial = -1.0
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.8, 0.2])
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array
    a = make_nan_array((10,), np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = False
    initial = 0.0
    where = np.ones((10,), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array with negative values
    a = make_nan_array((2, 2, 3, 3), np.float64) * -2.0
    axis = [1, 3]
    dtype = np.float64
    keepdims = False
    initial = 5.0
    where = np.random.choice([True, False], size=(2, 2, 3, 3), p=[0.95, 0.05])
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, broadcast compatible 'where'
    a = make_nan_array((5, 5), np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = True
    initial = -2.5
    where = np.array([[True], [False], [True], [False], [True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, reduce axis 1
    a = make_nan_array((3, 4, 5), np.float64)
    axis = [1]
    dtype = np.float64
    keepdims = False
    initial = 0.0
    where = np.random.choice([True, False], size=(3, 4, 5), p=[0.7, 0.3])
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, all-False mask
    a = make_nan_array((4, 4), np.float32)
    axis = [1]
    dtype = np.float32
    keepdims = True
    initial = 100.0
    where = np.zeros((4, 4), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array reducing all axes
    a = make_nan_array((2, 2, 2), np.float32)
    axis = [0, 1, 2]
    dtype = np.float32
    keepdims = True
    initial = 0.0
    where = np.ones((2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, large size
    a = make_nan_array((2, 3, 2, 4), np.float64)
    axis = [2]
    dtype = np.float64
    keepdims = True
    initial = -0.5
    where = np.random.choice([True, False], size=(2, 3, 2, 4), p=[0.85, 0.15])
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nansum_3"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nansum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nansum_3'.")


check_valid('jax.numpy.nansum', generated_inputs['jax.numpy.nansum_3'], lib="jax", suffix=3)
