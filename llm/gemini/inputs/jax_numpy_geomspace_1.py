
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def geomspace_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range with float32
    input_dict = {
        "start": np.array(1.0, dtype=np.float32),
        "stop": np.array(1000.0, dtype=np.float32),
        "num": 10,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array inputs, endpoint=False
    input_dict = {
        "start": np.array([1.0, 2.0], dtype=np.float32),
        "stop": np.array([10.0, 20.0], dtype=np.float32),
        "num": 5,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array inputs with float64
    input_dict = {
        "start": np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64),
        "stop": np.array([[10.0, 100.0], [1000.0, 10000.0]], dtype=np.float64),
        "num": 4,
        "endpoint": True,
        "dtype": np.float64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative range (geomspace works if signs are the same)
    input_dict = {
        "start": np.array([-1.0, -2.0], dtype=np.float32),
        "stop": np.array([-100.0, -200.0], dtype=np.float32),
        "num": 6,
        "endpoint": True,
        "dtype": np.float32,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional, changing axis to 1
    input_dict = {
        "start": np.ones((2, 3), dtype=np.float32) * 2.0,
        "stop": np.ones((2, 3), dtype=np.float32) * 2000.0,
        "num": 8,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex values
    input_dict = {
        "start": np.array([1.0 + 0j], dtype=np.complex64),
        "stop": np.array([100.0 + 0j], dtype=np.complex64),
        "num": 5,
        "endpoint": True,
        "dtype": np.complex64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small steps, high precision
    input_dict = {
        "start": np.array(0.001, dtype=np.float64),
        "stop": np.array(0.1, dtype=np.float64),
        "num": 3,
        "endpoint": True,
        "dtype": np.float64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large number of generated values, float32
    input_dict = {
        "start": np.array([1.0], dtype=np.float32),
        "stop": np.array([100000.0], dtype=np.float32),
        "num": 100,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor with axis=2
    input_dict = {
        "start": np.ones((2, 2, 2), dtype=np.float32) * 5.0,
        "stop": np.ones((2, 2, 2), dtype=np.float32) * 125.0,
        "num": 5,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional complex128
    input_dict = {
        "start": np.array([[1.0 + 1j, 2.0 + 2j]], dtype=np.complex128),
        "stop": np.array([[10.0 + 10j, 20.0 + 20j]], dtype=np.complex128),
        "num": 4,
        "endpoint": False,
        "dtype": np.complex128,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.geomspace_1"] = geomspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.geomspace_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.geomspace_1'.")


check_valid('jax.numpy.geomspace', generated_inputs['jax.numpy.geomspace_1'], lib="jax", suffix=1)
