
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmax_inputs():
    list_of_inputs = []
    
    # Input 1: 2D float32, axis 0, keepdims False
    a = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, -1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    where = np.array([[True, True, False], [False, True, True], [True, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "initial": np.array(-10.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32, axis 0, keepdims True
    a = np.array([np.nan, -2.0, 3.0, np.nan, 0.0], dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": True,
        "initial": np.array(-5.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, axis 1, keepdims False
    a = np.random.randn(2, 3, 4).astype(np.float64)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = np.random.choice([True, False], size=(2, 3, 4)).astype(bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": False,
        "initial": np.array(-100.0, dtype=np.float64),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, axis -1, keepdims True, negative values
    a = np.array([[-1.0, -2.0], [np.nan, -4.0]], dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "keepdims": True,
        "initial": np.array(-10.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32, axis 1, keepdims False
    a = np.random.randn(3, 2).astype(np.float32)
    where = np.ones((3, 2), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": False,
        "initial": np.array(0.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32, axis 2, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "keepdims": True,
        "initial": np.array(-10.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64, axis 0, keepdims False
    a = np.array([np.nan, np.nan, -10.0, -20.0, np.nan], dtype=np.float64)
    where = np.array([True, False, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "initial": np.array(-1000.0, dtype=np.float64),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 with dimension size 1, axis 2, keepdims True
    a = np.random.randn(3, 1, 4).astype(np.float32)
    where = np.ones((3, 1, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "keepdims": True,
        "initial": np.array(-50.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32, axis 0, keepdims False, random boolean mask
    a = np.random.randn(5, 5).astype(np.float32)
    where = np.random.choice([True, False], size=(5, 5)).astype(bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "initial": np.array(-20.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, axis -2, keepdims True
    a = np.random.randn(2, 4, 3).astype(np.float32)
    where = np.ones((2, 4, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": -2,
        "keepdims": True,
        "initial": np.array(-5.0, dtype=np.float32),
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanmax_1"] = nanmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmax_1'.")


check_valid('jax.numpy.nanmax', generated_inputs['jax.numpy.nanmax_1'], lib="jax", suffix=1)
