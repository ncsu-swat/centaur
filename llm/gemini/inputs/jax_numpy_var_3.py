
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def var_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, axis 0
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = 0
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64, axis 1
    a = np.array(
        [[-2.5, 3.1, -0.4], [1.2, -4.5, 2.3], [0.0, 1.5, -1.5]],
        dtype=np.float64,
    )
    axis = 1
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.array(
        [[True, True, False], [True, True, True], [False, True, True]]
    )
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32, axis 0
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    axis = 0
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.array([True, True, True, True, False])
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32, axis -1
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 10.0
    axis = -1
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64, axis 1
    a = np.linspace(-5.0, 5.0, 30, dtype=np.float64).reshape(2, 5, 3)
    axis = 1
    mean = np.mean(a, axis=axis, keepdims=True)
    where = a > 0.0
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32, axis 2
    a = np.ones((2, 3, 4, 2), dtype=np.float32) * 3.5
    axis = 2
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array promoted to float32, axis 0
    a = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    axis = 0
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 array, axis 0
    a = np.array([-1e4, 0.0, 1e4, 2e4], dtype=np.float64)
    axis = 0
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, negative axis -2
    a = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    axis = -2
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.ones((2, 2, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array, broadcasting where mask
    a = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    axis = 1
    mean = np.mean(a, axis=axis, keepdims=True)
    where = np.array([[True, True, True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.var_3"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_3'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_3'], lib="jax", suffix=3)
