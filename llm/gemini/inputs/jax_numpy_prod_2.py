
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def prod_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "a": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "axis": (0,),
        "dtype": np.int32,
        "keepdims": True,
        "initial": np.array(1, dtype=np.int32),
        "where": np.array([[True, False], [True, True]], dtype=bool),
        "promote_integers": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "a": np.random.randn(3, 4, 5).astype(np.float32),
        "axis": (1, 2),
        "dtype": np.float32,
        "keepdims": False,
        "initial": np.array(1.0, dtype=np.float32),
        "where": np.ones((3, 4, 5), dtype=bool),
        "promote_integers": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "a": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "axis": (0,),
        "dtype": np.int64,
        "keepdims": True,
        "initial": np.array(2, dtype=np.int64),
        "where": np.array([True, True, False, True, True], dtype=bool),
        "promote_integers": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "a": np.random.uniform(-2, 2, (2, 2, 2)).astype(np.float64),
        "axis": (0, 2),
        "dtype": np.float64,
        "keepdims": False,
        "initial": np.array(0.5, dtype=np.float64),
        "where": np.random.choice([True, False], size=(2, 2, 2)),
        "promote_integers": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "a": np.ones((5,), dtype=np.float32) * -1.0,
        "axis": (0,),
        "dtype": np.float32,
        "keepdims": True,
        "initial": np.array(-1.0, dtype=np.float32),
        "where": np.array([True, False, True, False, True], dtype=bool),
        "promote_integers": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "a": np.array([[10, 20], [30, 40]], dtype=np.int32),
        "axis": (1,),
        "dtype": np.int64,
        "keepdims": False,
        "initial": np.array(1, dtype=np.int64),
        "where": np.ones((2, 2), dtype=bool),
        "promote_integers": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "axis": (1,),
        "dtype": np.float32,
        "keepdims": True,
        "initial": np.array(1.0, dtype=np.float32),
        "where": np.ones((2, 3, 4), dtype=bool),
        "promote_integers": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "a": np.array([[[1.5]]], dtype=np.float32),
        "axis": (0, 1, 2),
        "dtype": np.float32,
        "keepdims": False,
        "initial": np.array(1.0, dtype=np.float32),
        "where": np.array([[[True]]], dtype=bool),
        "promote_integers": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "a": np.arange(1, 10).reshape(3, 3).astype(np.int32),
        "axis": (0,),
        "dtype": np.int32,
        "keepdims": False,
        "initial": np.array(10, dtype=np.int32),
        "where": np.array([[True, True, True], [False, True, False], [True, True, True]], dtype=bool),
        "promote_integers": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "a": np.random.randn(2, 2, 2).astype(np.float64),
        "axis": (2,),
        "dtype": np.float64,
        "keepdims": True,
        "initial": np.array(1.0, dtype=np.float64),
        "where": np.ones((2, 2, 2), dtype=bool),
        "promote_integers": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.prod_2"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_2'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_2'], lib="jax", suffix=2)
