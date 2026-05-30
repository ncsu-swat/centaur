
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def mean_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, axis 0, float32 output
    a = np.random.randn(4, 4).astype(np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    where = np.ones((4, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float64 array, axis 1, keepdims True, with a custom mask
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 1
    dtype = np.float64
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 4))
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array, axis 0, float32 output
    a = np.random.randint(-10, 10, size=(10,)).astype(np.int32)
    axis = 0
    dtype = np.float32
    keepdims = False
    where = np.ones((10,), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array, axis 2, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 2
    dtype = np.float32
    keepdims = True
    where = np.random.choice([True, False], size=(2, 2, 2, 2))
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, negative axis, float64 output
    a = np.random.randn(5, 3).astype(np.float32)
    axis = -1
    dtype = np.float64
    keepdims = False
    where = np.ones((5, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float16 array, axis 0, keepdims True, float32 output
    a = np.random.randn(8).astype(np.float16)
    axis = 0
    dtype = np.float32
    keepdims = True
    where = np.ones((8,), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int16 array, negative axis, float32 output
    a = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int16)
    axis = -2
    dtype = np.float32
    keepdims = False
    where = np.ones((3, 3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64 array, axis 1, float64 output, keepdims True
    a = np.random.randn(10, 10).astype(np.float64)
    axis = 1
    dtype = np.float64
    keepdims = True
    where = np.random.choice([True, False], size=(10, 10))
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 array, axis 3, float32 output
    a = np.random.randn(2, 1, 3, 2, 4).astype(np.float32)
    axis = 3
    dtype = np.float32
    keepdims = False
    where = np.ones((2, 1, 3, 2, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, axis 0, keepdims True, broadcasted where mask
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    dtype = np.float32
    keepdims = True
    where = np.ones((1, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.mean_1"] = mean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mean_1'.")


check_valid('jax.numpy.mean', generated_inputs['jax.numpy.mean_1'], lib="jax", suffix=1)
