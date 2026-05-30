
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sum_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "a": np.random.randn(3, 4).astype(np.float32),
        "axis": (0,),
        "dtype": np.float32,
        "keepdims": True,
        "initial": np.array(0.0, dtype=np.float32),
        "where": np.ones((3, 4), dtype=np.bool_),
        "promote_integers": True
    })

    # Input 2
    list_of_inputs.append({
        "a": np.random.randint(-10, 10, size=(5, 5, 5)).astype(np.int32),
        "axis": (1, 2),
        "dtype": np.int64,
        "keepdims": False,
        "initial": np.array(10, dtype=np.int64),
        "where": np.random.choice([True, False], size=(5, 5, 5)),
        "promote_integers": False
    })

    # Input 3
    list_of_inputs.append({
        "a": np.random.randn(2, 2).astype(np.float64),
        "axis": (0, 1),
        "dtype": np.float64,
        "keepdims": True,
        "initial": np.array(-1.5, dtype=np.float64),
        "where": np.array([[True, False], [True, True]], dtype=np.bool_),
        "promote_integers": True
    })

    # Input 4
    list_of_inputs.append({
        "a": np.random.randint(0, 5, size=(4,)).astype(np.int16),
        "axis": (0,),
        "dtype": np.int32,
        "keepdims": False,
        "initial": np.array(5, dtype=np.int32),
        "where": np.array([True, False, True, False], dtype=np.bool_),
        "promote_integers": True
    })

    # Input 5
    list_of_inputs.append({
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "axis": (0, 2),
        "dtype": np.float32,
        "keepdims": True,
        "initial": np.array(0.5, dtype=np.float32),
        "where": np.ones((2, 1, 4), dtype=np.bool_),
        "promote_integers": False
    })

    # Input 6
    list_of_inputs.append({
        "a": np.random.randn(6, 6).astype(np.float32),
        "axis": (1,),
        "dtype": np.float64,
        "keepdims": False,
        "initial": np.array(-0.0, dtype=np.float64),
        "where": np.ones((6, 6), dtype=np.bool_),
        "promote_integers": True
    })

    # Input 7
    list_of_inputs.append({
        "a": np.random.randint(-100, 100, size=(10,)).astype(np.int64),
        "axis": (0,),
        "dtype": np.int64,
        "keepdims": True,
        "initial": np.array(100, dtype=np.int64),
        "where": np.ones((10,), dtype=np.bool_),
        "promote_integers": False
    })

    # Input 8
    list_of_inputs.append({
        "a": np.random.randn(3, 3, 3, 3).astype(np.float32),
        "axis": (0, 1, 2),
        "dtype": np.float32,
        "keepdims": False,
        "initial": np.array(1.0, dtype=np.float32),
        "where": np.ones((3, 3, 3, 3), dtype=np.bool_),
        "promote_integers": True
    })

    # Input 9
    list_of_inputs.append({
        "a": np.random.randn(2, 4).astype(np.float32),
        "axis": (1,),
        "dtype": np.float32,
        "keepdims": True,
        "initial": np.array(0.0, dtype=np.float32),
        "where": np.array([[True, False, True, False], [False, True, False, True]], dtype=np.bool_),
        "promote_integers": True
    })

    # Input 10
    list_of_inputs.append({
        "a": np.random.randint(1, 10, size=(3, 2)).astype(np.int32),
        "axis": (0, 1),
        "dtype": np.int32,
        "keepdims": False,
        "initial": np.array(-5, dtype=np.int32),
        "where": np.array([[True, True], [False, False], [True, False]], dtype=np.bool_),
        "promote_integers": True
    })

    return list_of_inputs

generated_inputs["jax.numpy.sum_2"] = jax_numpy_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_2'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_2'], lib="jax", suffix=2)
