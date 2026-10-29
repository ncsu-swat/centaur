
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, custom positive spacing
    f = np.linspace(-5.0, 5.0, 20, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, gradient along axis 0
    f = np.random.randn(5, 6).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 2.0,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, gradient along axis 1
    f = np.random.randn(8, 12).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.25,
        "axis": 1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array, negative axis index
    f = np.random.randn(6, 7).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 1.5,
        "axis": -1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 0
    f = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.1,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, axis 1
    f = np.random.randn(4, 5, 6).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.05,
        "axis": 1,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, axis 2
    f = np.random.randn(4, 4, 8).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 3.0,
        "axis": 2,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, axis 3
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": 3,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64 array, negative axis index
    f = np.random.randn(2, 4, 3, 6).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": -2,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with trigonometric values
    f = np.sin(np.linspace(-np.pi, np.pi, 50)).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.01,
        "axis": 0,
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_4"] = gradient_inputs()

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
