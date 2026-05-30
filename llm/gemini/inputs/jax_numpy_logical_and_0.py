
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logical_and_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D boolean arrays of same shape
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D boolean arrays of same shape
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([[True, True], [False, False]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 1D integer arrays (logical and evaluates non-zero as True)
    x = np.array([0, 1, 2, 0], dtype=np.int32)
    y = np.array([1, 0, 3, 0], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Floating point arrays with negative, positive, and zero values
    x = np.array([-1.5, 0.0, 2.3], dtype=np.float32)
    y = np.array([0.0, 1.0, -0.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting a 2D array with a 1D array
    x = np.array([[True, False, True], [False, True, False]], dtype=bool)
    y = np.array([True, True, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Broadcasting with a 0D array (scalar-like tensor)
    x = np.array([True, False, True], dtype=bool)
    y = np.array(True, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Higher-dimensional boolean arrays (3D)
    x = np.random.choice([True, False], size=(2, 3, 4))
    y = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Empty arrays (preserving shape but size 0)
    x = np.empty((0, 5), dtype=bool)
    y = np.empty((0, 5), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Large 2D arrays of mixed integer types
    x = np.random.randint(-5, 5, size=(10, 10)).astype(np.int64)
    y = np.random.randint(-5, 5, size=(10, 10)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Float64 high precision arrays with broadcasting (1, 5) and (5, 1)
    x = np.random.randn(1, 5).astype(np.float64)
    y = np.random.randn(5, 1).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 4D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2, 2))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.logical_and"] = logical_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_and'.")


check_valid('jax.numpy.logical_and', generated_inputs['jax.numpy.logical_and'], lib="jax", suffix=0)
