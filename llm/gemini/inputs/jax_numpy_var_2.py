
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def var_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float32, default correction
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float64, multiple axes reduction
    a = np.random.randn(3, 5, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (1, 2),
        "dtype": np.dtype('float64'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D with where mask
    a = np.random.randn(3, 3).astype(np.float32)
    where = np.array([[True, False, True], [False, True, True], [True, True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D with precomputed mean (broadcast compatible)
    a = np.random.randn(4, 3).astype(np.float32)
    mean = np.mean(a, axis=(1,), keepdims=True)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": mean,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D with correction=1
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values, float64
    a = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "dtype": np.dtype('float64'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D with correction=2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (2,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D with where and precomputed mean
    a = np.random.randn(3, 4).astype(np.float32)
    where = np.array([[True, True, False, True], [False, True, True, True], [True, False, True, False]], dtype=bool)
    mean = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 3D tensor, multi-axis reduction
    a = np.random.randn(10, 10, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.var_2"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_2'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_2'], lib="jax", suffix=2)
