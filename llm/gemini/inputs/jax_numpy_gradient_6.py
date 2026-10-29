
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis (0,), float32
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis (0,), float64
    f = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
    )
    input_dict = {
        "f": f,
        "varargs": 0.5,
        "axis": (0,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis (1,), float32
    f = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 2.0,
        "axis": (1,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, multiple axes (0, 1), float64
    f = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0, 1),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, non-consecutive axes (0, 2), float32
    f = np.random.randn(4, 3, 5).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.25,
        "axis": (0, 2),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis (2,), float64
    f = np.random.randn(3, 4, 6).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 1.5,
        "axis": (2,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, negative values, float32
    f = -np.abs(np.random.randn(5, 5, 5)).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 0.1,
        "axis": (1,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axes (0, 1, 2), float32
    f = np.random.randn(3, 4, 3, 4).astype(np.float32)
    input_dict = {
        "f": f,
        "varargs": 1.0,
        "axis": (0, 1, 2),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axes (2, 3), float64
    f = np.random.randn(2, 3, 5, 5).astype(np.float64)
    input_dict = {
        "f": f,
        "varargs": 3.0,
        "axis": (2, 3),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with linear values and small spacing, float64
    f = np.linspace(-10.0, 10.0, 20, dtype=np.float64)
    input_dict = {
        "f": f,
        "varargs": 0.05,
        "axis": (0,),
        "edge_order": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.gradient_6"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_6'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_6'], lib="jax", suffix=6)
