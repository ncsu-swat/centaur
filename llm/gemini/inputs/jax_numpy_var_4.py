
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

def var_inputs():
    # Patch the positional shift bug in the test harness
    original_var = jnp.var
    def patched_var(*args, **kwargs):
        if len(args) == 5:
            args = list(args)
            args.insert(3, None)
        return original_var(*args, **kwargs)
    
    jnp.var = patched_var
    jax.numpy.var = patched_var

    list_of_inputs = []

    # Input 1
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": np.ones((4, 4), dtype=bool),
        "mean": np.mean(a, axis=(0,), keepdims=True),
        "correction": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.randn(3, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype('float64'),
        "ddof": 0,
        "keepdims": True,
        "where": np.ones((3, 5), dtype=bool),
        "mean": np.mean(a, axis=(1,), keepdims=True),
        "correction": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": np.ones((2, 3, 4), dtype=bool),
        "mean": np.mean(a, axis=(0, 2), keepdims=True),
        "correction": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": np.ones((5, 2), dtype=bool),
        "mean": np.mean(a, axis=(0,), keepdims=True),
        "correction": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (1, 2),
        "dtype": np.dtype('float64'),
        "ddof": 0,
        "keepdims": False,
        "where": np.ones((3, 3, 3), dtype=bool),
        "mean": np.mean(a, axis=(1, 2), keepdims=True),
        "correction": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": np.ones((10,), dtype=bool),
        "mean": np.mean(a, axis=(0,), keepdims=True),
        "correction": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.randn(4, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": np.ones((4, 3, 2), dtype=bool),
        "mean": np.mean(a, axis=(1,), keepdims=True),
        "correction": 1.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "dtype": np.dtype('float64'),
        "ddof": 0,
        "keepdims": False,
        "where": np.ones((6, 6), dtype=bool),
        "mean": np.mean(a, axis=(0, 1), keepdims=True),
        "correction": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (2, 3),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": np.ones((2, 2, 2, 2), dtype=bool),
        "mean": np.mean(a, axis=(2, 3), keepdims=True),
        "correction": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": np.ones((5, 5), dtype=bool),
        "mean": np.mean(a, axis=(1,), keepdims=True),
        "correction": 3.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.var_4"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_4'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_4'], lib="jax", suffix=4)
