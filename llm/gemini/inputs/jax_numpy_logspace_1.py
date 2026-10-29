
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logspace_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars as arrays, default base (10)
    input_dict = {
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(3.0, dtype=np.float32),
        "num": 10,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative range, endpoint False
    input_dict = {
        "start": np.array(-3.0, dtype=np.float64),
        "stop": np.array(-1.0, dtype=np.float64),
        "num": 5,
        "endpoint": False,
        "base": np.array(10.0, dtype=np.float64),
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Base 2 with integer inputs
    input_dict = {
        "start": np.array(1, dtype=np.int32),
        "stop": np.array(8, dtype=np.int32),
        "num": 8,
        "endpoint": True,
        "base": np.array(2, dtype=np.int32),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array inputs, axis=0
    input_dict = {
        "start": np.array([0.0, 1.0], dtype=np.float32),
        "stop": np.array([2.0, 3.0], dtype=np.float32),
        "num": 4,
        "endpoint": True,
        "base": np.array([10.0, 2.0], dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array inputs, axis=1
    input_dict = {
        "start": np.array([0.0, 2.0, 4.0], dtype=np.float32),
        "stop": np.array([1.0, 3.0, 5.0], dtype=np.float32),
        "num": 6,
        "endpoint": False,
        "base": np.array(2.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array inputs, axis=-1
    input_dict = {
        "start": np.array([-1.0, 0.0], dtype=np.float64),
        "stop": np.array([1.0, 2.0], dtype=np.float64),
        "num": 7,
        "endpoint": True,
        "base": np.array(np.e, dtype=np.float64),
        "dtype": np.dtype(np.float64),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array inputs, axis=0
    input_dict = {
        "start": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "stop": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array([[10.0, 10.0], [2.0, 2.0]], dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array inputs, axis=2
    input_dict = {
        "start": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "stop": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "num": 3,
        "endpoint": False,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex dtype
    input_dict = {
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(2.0, dtype=np.float32),
        "num": 4,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.dtype(np.complex64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array inputs, axis=1
    input_dict = {
        "start": np.zeros((2, 1, 3), dtype=np.float64),
        "stop": np.ones((2, 1, 3), dtype=np.float64),
        "num": 4,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float64),
        "dtype": np.dtype(np.float64),
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single element output (num=1)
    input_dict = {
        "start": np.array(2.0, dtype=np.float32),
        "stop": np.array(5.0, dtype=np.float32),
        "num": 1,
        "endpoint": True,
        "base": np.array(2.0, dtype=np.float32),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.logspace_1"] = logspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logspace_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logspace_1'.")


check_valid('jax.numpy.logspace', generated_inputs['jax.numpy.logspace_1'], lib="jax", suffix=1)
