
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_gradient_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, gradient along axis 1
    f = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": 1,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, gradient along axis 2
    f = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 2.0,
        "axis": 2,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative spacing, negative values
    f = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": -1.0,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, gradient along negative axis
    f = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.2,
        "axis": -1,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with float64 and very small spacing
    f = np.linspace(0, 1, 10, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.01,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, gradient along axis 0
    f = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.5,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D array with integer values (represented as float)
    f = np.arange(100, dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 10.0,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with large dimensions
    f = np.random.uniform(-100, 100, (50, 50)).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.1,
        "axis": 0,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, gradient along axis 3
    f = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.25,
        "axis": 3,
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_4"] = jax_numpy_gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_4'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_4'], lib="jax", suffix=4)
