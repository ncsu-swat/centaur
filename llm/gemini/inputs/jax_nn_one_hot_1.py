
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def one_hot_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, default-like axis=-1, float32 dtype
    x = np.array([0, 1, 2, 1], dtype=np.int32)
    num_classes = 3
    dtype = np.dtype(np.float32)
    axis = -1
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, axis=0, float64 dtype
    x = np.array([[0, 2, 4], [1, 3, 0]], dtype=np.int32)
    num_classes = 5
    dtype = np.dtype(np.float64)
    axis = 0
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int64 array with out-of-bounds indices, int32 dtype
    x = np.array([-1, 0, 2, 4, 10], dtype=np.int64)
    num_classes = 4
    dtype = np.dtype(np.int32)
    axis = 1
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array, axis=0, float32 dtype
    x = np.array(1, dtype=np.int32)
    num_classes = 2
    dtype = np.dtype(np.float32)
    axis = 0
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array, bool dtype, axis=2
    x = np.random.randint(0, 6, size=(2, 3, 4), dtype=np.int32)
    num_classes = 6
    dtype = np.dtype(np.bool_)
    axis = 2
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint8 array, float16 dtype, negative axis
    x = np.array([[1, 7], [3, 0], [2, 5]], dtype=np.uint8)
    num_classes = 8
    dtype = np.dtype(np.float16)
    axis = -1
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int16 array, uint8 dtype, axis=-2
    x = np.array([0, 5, 2, 8, 1, 9], dtype=np.int16)
    num_classes = 10
    dtype = np.dtype(np.uint8)
    axis = -2
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, float32 dtype, axis=-1
    x = np.random.randint(0, 4, size=(1, 2, 3), dtype=np.int32)
    num_classes = 4
    dtype = np.dtype(np.float32)
    axis = -1
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int64 array, num_classes=1, int64 dtype, axis=1
    x = np.array([[0, -1], [0, 0]], dtype=np.int64)
    num_classes = 1
    dtype = np.dtype(np.int64)
    axis = 1
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int32 array, axis=-3, float32 dtype
    x = np.random.randint(0, 3, size=(2, 1, 3, 2), dtype=np.int32)
    num_classes = 3
    dtype = np.dtype(np.float32)
    axis = -3
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int32 array, complex64 dtype, axis=0
    x = np.array([0, 2, 6], dtype=np.int32)
    num_classes = 7
    dtype = np.dtype(np.complex64)
    axis = 0
    input_dict = {
        "x": x,
        "num_classes": num_classes,
        "dtype": dtype,
        "axis": axis,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.nn.one_hot_1"] = one_hot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.one_hot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.one_hot_1'.")


check_valid('jax.nn.one_hot', generated_inputs['jax.nn.one_hot_1'], lib="jax", suffix=1)
