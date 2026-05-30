
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def allclose_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 arrays, identical
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 arrays with negative values, close under rtol
    a = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[-1.00001, -2.00002], [3.00003, 4.00004]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-04,
        "atol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 arrays with NaN, equal_nan=True
    a = np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    b = np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 arrays with NaN, equal_nan=False
    a = np.array([1.0, np.nan], dtype=np.float32)
    b = np.array([1.0, np.nan], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Inf and -Inf values
    a = np.array([np.inf, -np.inf], dtype=np.float32)
    b = np.array([np.inf, -np.inf], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large absolute tolerance (atol) to tolerate larger differences
    a = np.array([100.0, 200.0], dtype=np.float32)
    b = np.array([105.0, 195.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 10.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large relative tolerance (rtol) to tolerate percentage differences
    a = np.array([100.0, 200.0], dtype=np.float32)
    b = np.array([120.0, 240.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 0.25,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D scalar-like arrays
    a = np.array(1.0, dtype=np.float32)
    b = np.array(1.00001, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-04,
        "atol": 1e-04,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional 4D arrays
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    b = np.ones((2, 2, 2, 2), dtype=np.float32) + 1e-9
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Integer arrays
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.allclose"] = allclose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.allclose'.")


check_valid('jax.numpy.allclose', generated_inputs['jax.numpy.allclose'], lib="jax", suffix=0)
