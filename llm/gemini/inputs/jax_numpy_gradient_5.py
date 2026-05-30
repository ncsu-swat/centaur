
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_gradient_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array with unit-like spacing
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, gradient along both axes
    f = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 2.0,
        "axis": (0, 1),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with custom positive float spacing
    f = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": (1, 2),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array over a linear space
    f = np.linspace(-10, 10, 50).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.2,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with float64 precision
    f = np.random.randn(10, 10).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 1.5,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D flat/constant array with large spacing
    f = np.ones((4, 4, 4), dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 10.0,
        "axis": (0, 1, 2),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative spacing
    f = np.random.randn(8, 6).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": -1.0,
        "axis": (1,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-dimensional 4D array
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (2, 3),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Simple range 1D array with small spacing
    f = np.arange(10).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.1,
        "axis": (0,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, computing gradient along axis 1 only
    f = np.random.randn(6, 6).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 5.5,
        "axis": (1,),
        "edge_order": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_5"] = jax_numpy_gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_5'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_5'], lib="jax", suffix=5)
