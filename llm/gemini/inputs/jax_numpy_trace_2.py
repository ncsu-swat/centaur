
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trace_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 array, default axes (0, 1), offset 0
    input_dict = {
        "a": np.random.randn(4, 4).astype(np.float32),
        "offset": np.array(0, dtype=np.int32),
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular float64 array, positive offset
    input_dict = {
        "a": np.random.randn(5, 7).astype(np.float64),
        "offset": np.array(2, dtype=np.int64),
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular int32 array, negative offset
    input_dict = {
        "a": np.random.randint(-10, 10, size=(6, 4)).astype(np.int32),
        "offset": np.array(-1, dtype=np.int32),
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axes 1 and 2, offset 0
    input_dict = {
        "a": np.random.randn(3, 4, 4).astype(np.float32),
        "offset": np.array(0, dtype=np.int32),
        "axis1": 1,
        "axis2": 2,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axes 0 and 2, positive offset
    input_dict = {
        "a": np.random.randint(0, 100, size=(5, 2, 5)).astype(np.int64),
        "offset": np.array(1, dtype=np.int32),
        "axis1": 0,
        "axis2": 2,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, negative axes (-2, -1), offset 0
    input_dict = {
        "a": np.random.randn(2, 3, 4, 4).astype(np.float32),
        "offset": np.array(0, dtype=np.int32),
        "axis1": -2,
        "axis2": -1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex64 array, axes 0 and 3, offset -2
    input_dict = {
        "a": (
            np.random.randn(4, 2, 3, 4) + 1j * np.random.randn(4, 2, 3, 4)
        ).astype(np.complex64),
        "offset": np.array(-2, dtype=np.int32),
        "axis1": 0,
        "axis2": 3,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array converted to int32 dtype
    input_dict = {
        "a": np.random.choice([True, False], size=(5, 5)),
        "offset": np.array(0, dtype=np.int32),
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float64 array, axes 2 and 4, offset 1
    input_dict = {
        "a": np.random.randn(2, 2, 3, 2, 3).astype(np.float64),
        "offset": np.array(1, dtype=np.int64),
        "axis1": 2,
        "axis2": 4,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, swapped axes (2, 0), negative offset
    input_dict = {
        "a": np.random.randn(4, 3, 4).astype(np.float32),
        "offset": np.array(-1, dtype=np.int32),
        "axis1": 2,
        "axis2": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.trace_2"] = trace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trace_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trace_2'.")


check_valid('jax.numpy.trace', generated_inputs['jax.numpy.trace_2'], lib="jax", suffix=2)
