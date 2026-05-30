
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logspace_inputs():
    list_of_inputs = []

    # Input 1: Basic logspace with default base 10
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(2.0, dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 2: Excluding endpoint
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(2.0, dtype=np.float32),
        "num": 5,
        "endpoint": False,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 3: Negative bounds and float64 dtype
    list_of_inputs.append({
        "start": np.array(-2.0, dtype=np.float64),
        "stop": np.array(-1.0, dtype=np.float64),
        "num": 10,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float64),
        "dtype": np.float64,
        "axis": 0
    })

    # Input 4: Different base (base 2)
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(4.0, dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array(2.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 5: 1D array boundaries for multidimensional logspace
    list_of_inputs.append({
        "start": np.array([0.0, 1.0], dtype=np.float32),
        "stop": np.array([2.0, 3.0], dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 6: 1D array boundaries with 1D array base
    list_of_inputs.append({
        "start": np.array([0.0, 1.0], dtype=np.float32),
        "stop": np.array([2.0, 3.0], dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array([10.0, 2.0], dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 7: Multi-dimensional boundaries with axis=-1
    list_of_inputs.append({
        "start": np.array([0.0, 1.0], dtype=np.float32),
        "stop": np.array([2.0, 3.0], dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": -1
    })

    # Input 8: 2D boundaries
    list_of_inputs.append({
        "start": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "stop": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "num": 4,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.float32,
        "axis": 0
    })

    # Input 9: Complex output dtype
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(2.0, dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "base": np.array(10.0, dtype=np.float32),
        "dtype": np.complex64,
        "axis": 0
    })

    # Input 10: Integer inputs producing float32 output
    list_of_inputs.append({
        "start": np.array(1, dtype=np.int32),
        "stop": np.array(3, dtype=np.int32),
        "num": 3,
        "endpoint": True,
        "base": np.array(10, dtype=np.int32),
        "dtype": np.float32,
        "axis": 0
    })

    return [copy.deepcopy(inp) for inp in list_of_inputs]

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
